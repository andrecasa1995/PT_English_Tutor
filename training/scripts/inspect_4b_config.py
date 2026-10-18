"""Inspect official small configuration files, without model weights."""
import json
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
LOG='/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log'
def log(s):
    print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')
repo='Qwen/Qwen3.5-4B'
log('[step 115] Inspecting Qwen3.5-4B model configuration')
try:
    path=hf_hub_download(repo,'config.json')
    with open(path,encoding='utf-8') as f: cfg=json.load(f)
    log('model_type: '+str(cfg.get('model_type')))
    log('architectures: '+str(cfg.get('architectures')))
    t=cfg.get('text_config',{})
    log('text_config model_type: '+str(t.get('model_type')))
    log('text hidden_size: '+str(t.get('hidden_size')))
    log('text num_hidden_layers: '+str(t.get('num_hidden_layers')))
    config=AutoConfig.from_pretrained(repo,local_files_only=True)
    log('AutoConfig class: '+type(config).__name__)
except Exception as e:
    log('CONFIG CHECK FAILED: '+type(e).__name__+': '+str(e))
    raise
log('[step 115] Configuration inspection complete; no weights downloaded')
