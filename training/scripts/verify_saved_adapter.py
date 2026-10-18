from pathlib import Path
from safetensors import safe_open
p=Path('/mnt/c/Users/andre/Documents/PT_English_Tutor/training/outputs/experimental_qwen35_4b_lora_2steps')
f=p/'adapter_model.safetensors'
assert f.exists() and f.stat().st_size>0
with safe_open(str(f),framework='pt',device='cpu') as sf:
    n=len(sf.keys())
assert n>0
assert (p/'adapter_config.json').exists()
assert (p/'experiment_metadata.json').exists()
print(f'[step 123] Saved adapter verified: {f.stat().st_size:,} bytes, {n} tensors')
print('[step 123] CHECKPOINT VALIDATION: SUCCESS')
