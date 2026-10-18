"""Controlled text-only Qwen3.5-4B 4-bit loading trial; no training or model changes."""
import sys, time, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL='Qwen/Qwen3.5-4B'
def log(s): print('[step 117] '+str(s),flush=True)
try:
 log('Starting quantized text-only model loading')
 free,total=torch.cuda.mem_get_info()
 log(f'VRAM free at start: {free/1024**3:.2f}/{total/1024**3:.2f} GiB')
 if free < 5 * 1024**3: raise RuntimeError('Less than 5 GiB VRAM free: aborting for safety')
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
 start=time.monotonic()
 tokenizer=AutoTokenizer.from_pretrained(MODEL)
 log('Tokenizer loaded; downloading/loading model weights as needed')
 model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=q,device_map={'':0},dtype=torch.float16,low_cpu_mem_usage=True)
 torch.cuda.synchronize()
 log(f'Model class: {type(model).__name__}')
 log(f'Quantized: {getattr(model,"is_loaded_in_4bit",False)}')
 log(f'PyTorch allocated GiB: {torch.cuda.memory_allocated()/1024**3:.3f}')
 log(f'PyTorch peak reserved GiB: {torch.cuda.max_memory_reserved()/1024**3:.3f}')
 free,_=torch.cuda.mem_get_info()
 log(f'Driver free VRAM after loading GiB: {free/1024**3:.3f}')
 log(f'Elapsed seconds: {time.monotonic()-start:.1f}')
 log('QWEN3.5-4B QUANTIZED LOADING: SUCCESS')
except Exception as exc:
 log(f'LOAD FAILED: {type(exc).__name__}: {exc}')
 traceback.print_exc()
 sys.exit(1)
