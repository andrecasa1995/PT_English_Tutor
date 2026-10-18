"""Inspect model class routing without weight downloads."""
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM
LOG='/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log'
def log(s):
 print(s,flush=True)
 with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')
log('[step 116] Checking Qwen3.5 model class support')
c=AutoConfig.from_pretrained('Qwen/Qwen3.5-4B',local_files_only=True)
for name, factory, config in [('CausalLM full config',AutoModelForCausalLM,c),('CausalLM text config',AutoModelForCausalLM,c.text_config),('Multimodal full config',AutoModelForMultimodalLM,c)]:
 try:
  cls=factory._model_mapping[type(config)]
  log(f'{name}: {cls.__name__}')
 except Exception as e: log(f'{name}: NOT MAPPED ({type(e).__name__})')
log('[step 116] Completed; 4B weights have not been downloaded')
