"""Non-destructive model preflight, without downloading model weights."""
import os, shutil
import torch
from huggingface_hub import HfApi

LOG = '/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log'
def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(msg+'\n')

log('[step 114] Qwen3.5-4B preflight starting (metadata only)')
log(f'CUDA: {torch.cuda.is_available()}')
free, total = torch.cuda.mem_get_info()
log(f'GPU free/total: {free/1024**3:.2f} / {total/1024**3:.2f} GiB')
for p in ('/home/andre', '/mnt/c/Users/andre/Documents/PT_English_Tutor'):
    usage = shutil.disk_usage(p)
    log(f'Disk {p}: {usage.free/1024**3:.1f} GiB free')
repo = 'Qwen/Qwen3.5-4B'
try:
    files = HfApi().list_repo_files(repo)
    weights = [f for f in files if f.endswith('.safetensors')]
    log(f'Official repo: {repo}; safetensors files: {len(weights)}')
    for f in weights[:8]: log('  '+f)
except Exception as e:
    log(f'Metadata request failed: {type(e).__name__}: {e}')
log('[step 114] Preflight finished; no large weights downloaded')
