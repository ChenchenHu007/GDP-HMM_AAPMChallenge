import torch
import torch.nn as nn
import torch.optim as optim
from nnunet_mednext import create_mednext_v1
import data_loader
# import data_loader_cache as data_loader
import yaml
import argparse 
import os
import time
from toolkit import print_log_to_file
'''
In this script, we provide a basic (and simple) pipeline designed for successful execution.
There are numerous advanced AI methodologies and strategies that could potentially improve the model's performance. 
We encourage participants to explore these AI technologies independently. The organizers will not provide much support for these explorations.
Please note that discussions/questions about AI tech explorations are not supposed to be raised in the repository issues.

Reminder: The information provided in the meta files is crucial, as it directly impacts how the reference is created. 
An example of how to use these information are provided in the data_loader.py. 
If you have questions related to clinical backgrounds, feel free to start a discussion.
'''

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--cfig_path',  type = str, default='config_files/config-amp.yaml')
parser.add_argument('--phase', default = 'train', type = str)
parser.add_argument('--use_amp', default = True, type = bool)
args = parser.parse_args()

cfig = yaml.load(open(args.cfig_path, 'r'), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ data loader -----------------#
loaders = data_loader.GetLoader(cfig = cfig['loader_params'])
prepare_time = time.perf_counter()
train_loader =loaders.train_dataloader()
# val_loader = loaders.val_dataloader()
prepare_time = time.perf_counter() - prepare_time
print(f"Data preparation time: {prepare_time:.2f}")
print_log_to_file(output_dst='/ai/aapm/GDP-HMM_AAPMChallenge/lightning/amp.txt',
                  txt=f'Data preparation time: {prepare_time:.2f}\n', 
                  mode='a')
del prepare_time

# ------------- Network ------------------ # 
model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
  num_classes = cfig['model_params']['out_channels'],
  model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
  kernel_size = cfig['model_params']['kernel_size'],   # 3x3x3 and 5x5x5 were tested in publication
  deep_supervision = cfig['model_params']['deep_supervision']   
).to(device)

# ------------ loss -----------------------# 
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr':cfig['lr']}], lr=cfig['lr'])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= len(train_loader) * cfig['num_epochs'], last_epoch=cfig['num_epochs'])
scaler = torch.amp.GradScaler(device, enabled=args.use_amp)

criterion = nn.L1Loss()

# -----------Training loop --------------- #

# nbatch_per_log = max(int(len(train_loader) / 20), 1)
nbatch_per_log = 100

#todo: validation loop
for epoch in range(cfig['num_epochs']):
    model.train()
    epoch_loss = 0

    train_time_per_epoch = 0
    train_loader_time_per_epoch = 0
    time_start_train = time.perf_counter()
    time_start_load_data = time.perf_counter()

    for batch_idx, data_dict in enumerate(train_loader):
        # Forward pass
        data = data_dict['data'].to(device)
        label = data_dict['label'].to(device)
        train_loader_time_per_epoch += time.perf_counter() - time_start_load_data

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            outputs = model(data)
            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone()) 

            outputs = outputs * cfig['scale_out']
            loss = criterion(outputs, label)
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()

        epoch_loss += loss.item()
        
        if batch_idx % nbatch_per_log == 0:
        
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{cfig['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

        # if batch_idx == 3 and epoch % 50 == 0:
            # os.system('nvidia-smi')

        time_start_load_data = time.perf_counter()

    train_time_per_epoch = time.perf_counter() - time_start_train
    print_log_to_file(output_dst='/ai/aapm/GDP-HMM_AAPMChallenge/lightning/amp.txt', 
                      txt=f'Epoch {epoch+1}:,\nTrain Time: {train_time_per_epoch:.2f},\nData Load Time: {train_loader_time_per_epoch:.2f}\n', 
                      mode='a')
    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{cfig['num_epochs']}] Completed: Avg Loss: {avg_epoch_loss:.4f}")
    print_log_to_file(output_dst='/ai/aapm/GDP-HMM_AAPMChallenge/lightning/amp.txt', 
                      txt=f'Avg Loss: {avg_epoch_loss:.4f}\n\n', 
                      mode='a')
    # Save the model
    model_save_path = os.path.join(cfig['save_model_root'], 'last_model.pth')
    torch.save(model.state_dict(), model_save_path)
