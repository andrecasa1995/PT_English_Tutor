"""One supervised QLoRA smoke step on training JSONL; no checkpoint saved."""
import json, sys, traceback, torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
ROOT=Path(__file__).resolve().parents[2]
def log(v): print('[step 121] '+str(v),flush=True)
try:
    free,_=torch.cuda.mem_get_info()
    if free < 5*1024**3: raise RuntimeError('Not enough free GPU memory')
    example=json.loads((ROOT/'training/data/sft_train.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert len(example['messages'])==3
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',local_files_only=True)
    msgs=example['messages']
    prompt=tokenizer.apply_chat_template(msgs[:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(prompt): raise RuntimeError('Chat template mismatch: cannot safely mask prompt')
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    encoded=tokenizer(full,add_special_tokens=False)['input_ids']
    if encoded[:len(prefix)]!=prefix: raise RuntimeError('Token prefix mismatch')
    if len(encoded)>160: raise RuntimeError(f'Example too long ({len(encoded)} tokens), refusing truncation')
    input_ids=torch.tensor([encoded],device='cuda')
    labels=input_ids.clone(); labels[:,:len(prefix)]=-100
    supervision=(labels!=-100).sum().item()
    if supervision<2: raise RuntimeError('No target tokens')
    log(f'Tokens total={len(encoded)}, prompt={len(prefix)}, assistant target={supervision}')
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',quantization_config=q,device_map={'':0},dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False;model.train()
    optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    optim.zero_grad(set_to_none=True)
    loss=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),labels=labels).loss
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
    log(f'Supervised loss={loss.item():.5f}')
    loss.backward()
    count=sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    if not count: raise RuntimeError('No LoRA gradients')
    optim.step();torch.cuda.synchronize()
    log(f'Gradient tensors={count}; peak reserved VRAM={torch.cuda.max_memory_reserved()/1024**3:.3f} GiB')
    log('SUPERVISED QLORA ONE STEP SUCCESS; no adapter saved')
except BaseException as e:
    log(f'TEST FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()
    sys.exit(1)
