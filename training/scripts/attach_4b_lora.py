"""Attach QLoRA to cached Qwen3.5-4B; no training, saving or pushing."""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def log(s): print('[step 118] '+str(s),flush=True)
free,total=torch.cuda.mem_get_info()
log(f'Free VRAM before loading: {free/1024**3:.3f} GiB')
if free < 5*1024**3: raise RuntimeError('Insufficient free VRAM (<5 GiB)')
q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
log('Loading Qwen3.5-4B from local cache')
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',quantization_config=q,device_map={'':0},dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True)
log('Preparing k-bit model and attaching LoRA to attention projections')
model=prepare_model_for_kbit_training(model)
model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0.05,bias='none',task_type='CAUSAL_LM'))
model.print_trainable_parameters()
trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable>0
free,_=torch.cuda.mem_get_info()
log(f'Trainable parameters: {trainable:,}')
log(f'Allocated VRAM: {torch.cuda.memory_allocated()/1024**3:.3f} GiB')
log(f'Peak reserved VRAM: {torch.cuda.max_memory_reserved()/1024**3:.3f} GiB')
log(f'Free VRAM: {free/1024**3:.3f} GiB')
log('QLORA ADAPTER ATTACHMENT: SUCCESS (no training performed)')
