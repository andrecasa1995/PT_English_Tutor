"""Test loading experimental LoRA adapter with the cached NF4 base model."""
import torch,traceback,sys
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
p=Path('/mnt/c/Users/andre/Documents/PT_English_Tutor/training/outputs/experimental_qwen35_4b_lora_2steps')
def log(s):print('[step 124] '+str(s),flush=True)
try:
 free,_=torch.cuda.mem_get_info()
 if free<5*1024**3:raise RuntimeError('Less than 5 GiB free')
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
 base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',quantization_config=quant,device_map={'':0},dtype=torch.float16,local_files_only=True)
 model=PeftModel.from_pretrained(base,str(p),is_trainable=False)
 assert model.peft_config and len(model.peft_config)==1
 log(f'Adapter loaded: {list(model.peft_config.keys())}')
 log(f'Allocated VRAM: {torch.cuda.memory_allocated()/1024**3:.3f} GiB')
 log('ADAPTER RELOAD: SUCCESS')
except BaseException as e:
 log(f'RELOAD FAILED: {type(e).__name__}: {e}')
 traceback.print_exc();sys.exit(1)
