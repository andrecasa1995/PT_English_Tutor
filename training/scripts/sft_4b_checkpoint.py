"""Two-step QLoRA integration run on training examples only; saves experimental adapter."""
import json, torch, sys, traceback
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'training/data/sft_train.jsonl'
OUT=ROOT/'training/outputs/experimental_qwen35_4b_lora_2steps'
def log(s): print('[step 122] '+str(s),flush=True)
try:
 torch.manual_seed(42)
 free,_=torch.cuda.mem_get_info()
 if free<5*1024**3: raise RuntimeError('VRAM free below 5 GiB')
 records=[json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
 assert len(records)==12, 'Expected 12 training samples'
 assert not OUT.exists(), 'Experimental output exists; refusing overwrite'
 model_id='Qwen/Qwen3.5-4B'
 tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained(model_id,quantization_config=q,device_map={'':0},dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True)
 model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
 model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0,bias='none',task_type='CAUSAL_LM'))
 model.config.use_cache=False;model.train()
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
 losses=[]
 for i, rec in enumerate(records[:2],1):
  messages=rec['messages']
  prompt=tok.apply_chat_template(messages[:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  full=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
  assert full.startswith(prompt), 'Chat template mismatch'
  prefix=tok(prompt,add_special_tokens=False)['input_ids']
  tokens=tok(full,add_special_tokens=False)['input_ids']
  assert tokens[:len(prefix)]==prefix and len(tokens)<=192, 'Invalid length or token prefix'
  ids=torch.tensor([tokens],device='cuda'); labels=ids.clone(); labels[:,:len(prefix)]=-100
  assert (labels!=-100).sum().item()>1
  opt.zero_grad(set_to_none=True)
  loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
  assert torch.isfinite(loss), 'Loss not finite'
  loss.backward()
  assert any(p.grad is not None for p in model.parameters() if p.requires_grad),'No gradients'
  opt.step();torch.cuda.synchronize()
  losses.append(round(loss.item(),5))
  log(f'Step {i}/2 done; tokens={len(tokens)}, loss={losses[-1]}, peak={torch.cuda.max_memory_reserved()/1024**3:.3f} GiB')
 OUT.mkdir(parents=True,exist_ok=False)
 model.save_pretrained(str(OUT),safe_serialization=True)
 tok.save_pretrained(str(OUT))
 (OUT/'experiment_metadata.json').write_text(json.dumps({'base_model':model_id,'training_examples':2,'total_available_training_examples':len(records),'heldout_used':False,'losses':losses,'warning':'Technical smoke test only; not validated for deployment.'},indent=2),encoding='utf-8')
 log(f'Adapter saved to {OUT}')
 log('TWO-STEP SFT + CHECKPOINT: SUCCESS')
except BaseException as e:
 log(f'FAILED: {type(e).__name__}: {e}')
 traceback.print_exc();sys.exit(1)
