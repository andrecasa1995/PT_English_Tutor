"""One conservative Qwen3.5-4B NF4+LoRA training step. No weights saved."""
import gc, sys, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def log(s): print('[step 120] '+str(s), flush=True)
try:
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info()
    log(f'GPU free at start: {free/1024**3:.3f} GiB')
    if free < 5*1024**3: raise RuntimeError('Not enough VRAM for safe startup')
    torch.manual_seed(42)
    model_id='Qwen/Qwen3.5-4B'
    tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    log('Loading cached NF4 model')
    model=AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quant,device_map={'':0},dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0.0,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    model.train()
    log(f'Model and LoRA ready. GPU allocated: {torch.cuda.memory_allocated()/1024**3:.3f} GiB')
    tokens=tokenizer('Correct the English: She go to school every day.',return_tensors='pt',max_length=32,truncation=True).to('cuda')
    log(f'Sequence tokens: {tokens["input_ids"].shape[1]}')
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss=model(**tokens,labels=tokens['input_ids']).loss
    log(f'Loss: {loss.item():.5f}')
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads): raise RuntimeError('Missing or nonfinite gradients')
    optimizer.step();torch.cuda.synchronize()
    log(f'Gradient tensors: {len(grads)}')
    log(f'Peak PyTorch reserved: {torch.cuda.max_memory_reserved()/1024**3:.3f} GiB')
    log(f'Free GPU after step: {torch.cuda.mem_get_info()[0]/1024**3:.3f} GiB')
    log('QLORA 4B ONE TRAINING STEP: SUCCESS')
except BaseException as e:
    log(f'TEST FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()
    sys.exit(1)
