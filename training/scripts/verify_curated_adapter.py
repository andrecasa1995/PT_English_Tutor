"""Validate the experimental 8-step adapter checkpoint files."""
import json
from pathlib import Path
from safetensors import safe_open
p=Path(__file__).resolve().parents[1]/'outputs/experimental_qwen35_4b_curated_8steps'
f=p/'adapter_model.safetensors'
assert f.is_file() and f.stat().st_size>0
with safe_open(str(f),framework='pt',device='cpu') as sf:
 keys=sf.keys()
 assert len(keys)>0
 assert all(sf.get_tensor(key).isfinite().all().item() for key in keys)
meta=json.loads((p/'experiment_metadata.json').read_text(encoding='utf-8'))
assert len(meta['steps'])==8 and meta['heldout_used'] is False
assert (p/'adapter_config.json').exists()
print(f'[step 136] 8-step adapter verified: {f.stat().st_size:,} bytes / {len(keys)} finite tensors')
print('[step 136] CHECKPOINT VALID: SUCCESS')
