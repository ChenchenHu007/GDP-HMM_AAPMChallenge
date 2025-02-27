from huggingface_hub import snapshot_download
import os

if  __name__ == '__main__':
    os.environ['HF_HOME'] = '/ai'   # export HF

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # export HF_ENDPOINT=https://hf-mirror.com

    hf_token = 'your_hf_token'  # export HF_HOME=hf_token
    
    snapshot_download(repo_id="Jungle15/GDP-HMM_Challenge",  repo_type="dataset", 
                      local_dir='/ai', 
                      use_auth_token=hf_token)
    