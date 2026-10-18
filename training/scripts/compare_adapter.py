"""Bounded baseline-vs-adapter generation on one heldout input; no changes or training."""
import json,sys,traceback,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel
ROOT=Path('/mnt/c/Users/andre/Documents/PT_English_Tutor')
def log(v): print('[step 125] '+str(v),flush=True)
try:
 ex=json.loads((ROOT/'training/datasets/heldout_eval.json').read_text(encoding='utf-8'))['examples'][0]
 tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',local_files_only=True)
 from importlib.util import spec_from_file_location,module_from_spec
 spec=spec_from_file_location('prep',str(ROOT/'training/scripts/prepare_sft.py'));mod=module_from_spec(spec);spec.loader.exec_module(mod)
 prompt=tokenizer.apply_chat_template([{'role':'system','content':mod.SYSTEM},{'role':'user','content':ex['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 tokens=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
 log('Starting baseline vs adapter on 1 held-out sample (not used for training)')
 free,_=torch.cuda.mem_get_info()
 if free<5*1024**3: raise RuntimeError('Less than 5 GiB free GPU memory')
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',quantization_config=q,device_map={'':0},dtype=torch.float16,local_files_only=True)
 model.eval();model.config.use_cache=True
 def run(m):
  with torch.inference_mode(): output=m.generate(**tokens,max_new_tokens=100,do_sample=False,pad_token_id=tokenizer.eos_token_id)
  return tokenizer.decode(output[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True)
 log('Generating baseline')
 baseline=run(model)
 log('Loading saved adapter')
 trained=PeftModel.from_pretrained(model,str(ROOT/'training/outputs/experimental_qwen35_4b_lora_2steps'),is_trainable=False)
 trained.eval()
 log('Generating adapter response')
 adapted=run(trained)
 result={'sample_id':ex['id'],'input':ex['text'],'expected_edits':ex['edits'],'baseline_response':baseline,'adapter_response':adapted,'note':'Exploratory single-sample test; no performance conclusions.'}
 out=ROOT/'training/outputs/adapter_comparison_1sample.json'
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 log('Baseline: '+baseline[:450].replace('\n',' '))
 log('Adapter: '+adapted[:450].replace('\n',' '))
 log('Saved comparison to '+str(out))
 log('COMPARISON SUCCESS')
except BaseException as e:
 log(f'COMPARISON FAILED: {type(e).__name__}: {e}')
 traceback.print_exc();sys.exit(1)
