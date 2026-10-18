"""Run frozen held-out comparison. No training; save raw results and exact metrics."""
import json,sys,traceback,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel
R=Path('/mnt/c/Users/andre/Documents/PT_English_Tutor')
def log(s):print('[step 127] '+str(s),flush=True)
def parse(s):
 try:
  d=json.loads(s);return d if isinstance(d,dict) and isinstance(d.get('edits'),list) else None
 except (ValueError,TypeError):return None
def normalized(e):return sorted((v['original'],v['replacement'],v['category']) for v in e)
try:
 rows=json.loads((R/'training/datasets/heldout_eval.json').read_text(encoding='utf-8'))['examples']
 assert len(rows)==8
 if torch.cuda.mem_get_info()[0]<5*1024**3:raise RuntimeError('Less than 5 GiB free VRAM')
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',local_files_only=True)
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',quantization_config=q,device_map={'':0},dtype=torch.float16,local_files_only=True)
 adapter=PeftModel.from_pretrained(model,str(R/'training/outputs/experimental_qwen35_4b_lora_2steps'),is_trainable=False)
 adapter.eval()
 sysmsg='You are a careful American English writing tutor for Brazilian Portuguese speakers. Return JSON with a single edits array. Each edit has original, replacement, category, and explanation_pt. Only propose necessary, minimal edits; explain in Brazilian Portuguese.'
 output=[]
 for i,row in enumerate(rows,1):
  prompt=tok.apply_chat_template([{'role':'system','content':sysmsg},{'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inputs=tok(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
  for name,enabled in [('baseline',True),('adapter',False)]:
   with adapter.disable_adapter() if enabled else torch.no_grad():
    with torch.inference_mode(): ids=adapter.generate(**inputs,max_new_tokens=135,do_sample=False,pad_token_id=tok.eos_token_id)
   raw=tok.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
   result=parse(raw)
   try: exact= result is not None and normalized(result['edits'])==normalized(row['edits'])
   except (KeyError,TypeError,ValueError):exact=False
   output.append({'id':row['id'],'variant':name,'exact':exact,'valid_json':result is not None,'raw':raw})
   log(f'{i}/8 {row["id"]} {name}: valid_json={result is not None}; exact_edits={exact}')
  out=R/'training/outputs/heldout_comparison_8samples.json'
  out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
 for name in ('baseline','adapter'):
  rows2=[r for r in output if r['variant']==name]
  log(f'{name} exact={sum(r["exact"] for r in rows2)}/8; valid_json={sum(r["valid_json"] for r in rows2)}/8')
 log('HELDOUT COMPARISON COMPLETE')
except BaseException as e:
 log(f'FAILED: {type(e).__name__}: {e}')
 traceback.print_exc();sys.exit(1)
