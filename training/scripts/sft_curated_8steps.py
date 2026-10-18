"""Bounded 8-step synthetic SFT experiment; no evaluation data loaded."""
import json,torch,traceback,sys
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training/outputs/experimental_qwen35_4b_curated_8steps'
def log(x):print('[step 135] '+str(x),flush=True)
try:
 torch.manual_seed(42)
 if OUT.exists():raise RuntimeError('Checkpoint already exists; refusing overwrite')
 if torch.cuda.mem_get_info()[0]<5*1024**3:raise RuntimeError('Less than 5 GiB free GPU memory')
 rows=[json.loads(x) for x in (ROOT/'training/data/sft_curated_v1.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(rows)==49
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',local_files_only=True)
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',quantization_config=quant,device_map={'':0},dtype=torch.float16,local_files_only=True)
 model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
 model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0,bias='none',task_type='CAUSAL_LM'))
 model.train();model.config.use_cache=False
 optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=5e-5)
 metrics=[]
 for n,index in enumerate((0,3,10,15,20,27,36,45),1):
  messages=rows[index]['messages']
  prompt=tok.apply_chat_template(messages[:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  full=tok.apply_chat_template(messages,tokenize=False,enable_thinking=False)
  assert full.startswith(prompt)
  prefix=tok(prompt,add_special_tokens=False)['input_ids']; encoded=tok(full,add_special_tokens=False)['input_ids']
  assert encoded[:len(prefix)]==prefix and len(encoded)<=192,'Invalid token prefix or length'
  ids=torch.tensor([encoded],device='cuda');labels=ids.clone();labels[:,:len(prefix)]=-100
  assert (labels!=-100).sum().item()>0
  optimizer.zero_grad(set_to_none=True)
  loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
  assert torch.isfinite(loss).item()
  loss.backward()
  assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
  optimizer.step();torch.cuda.synchronize()
  metrics.append({'step':n,'index':index,'tokens':len(encoded),'loss':round(loss.item(),5)})
  log(f'Step {n}/8, sample {index}, tokens {len(encoded)}, loss {loss.item():.5f}, peak VRAM {torch.cuda.max_memory_reserved()/1024**3:.3f} GiB')
 OUT.mkdir(parents=True,exist_ok=False)
 model.save_pretrained(str(OUT),safe_serialization=True)
 tok.save_pretrained(str(OUT))
 (OUT/'experiment_metadata.json').write_text(json.dumps({'base_model':'Qwen/Qwen3.5-4B','dataset':'sft_curated_v1.jsonl','training_examples_total':len(rows),'steps':metrics,'heldout_used':False,'status':'Experimental, not validated for deployment'},indent=2),encoding='utf-8')
 log(f'Saved adapter {OUT}')
 log('EIGHT STEP TRAINING COMPLETE')
except BaseException as e:
 log(f'FAILED: {type(e).__name__}: {e}')
 traceback.print_exc();sys.exit(1)
