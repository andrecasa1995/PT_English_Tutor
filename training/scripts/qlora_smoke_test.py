"""One-step local QLoRA smoke test; does not save or publish model weights."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "Qwen/Qwen3-0.6B"
assert torch.cuda.is_available(), "CUDA unavailable"
print("Loading cached model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map={"": 0}, dtype=torch.float16, local_files_only=True)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
model.train()
model.config.use_cache = False
inputs = tokenizer("Correct the English sentence: She go to school every day.", return_tensors="pt", truncation=True, max_length=128).to("cuda")
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
optimizer.zero_grad(set_to_none=True)
loss = model(**inputs, labels=inputs["input_ids"]).loss
print("Loss:", round(loss.item(), 5), flush=True)
assert torch.isfinite(loss), "Nonfinite loss"
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert grads, "No LoRA gradients"
assert all(torch.isfinite(g).all().item() for g in grads), "Nonfinite gradients"
optimizer.step()
torch.cuda.synchronize()
print("LoRA gradient tensors:", len(grads))
print("Allocated GiB:", round(torch.cuda.memory_allocated()/1024**3, 3))
print("Peak reserved GiB:", round(torch.cuda.max_memory_reserved()/1024**3, 3))
print("FIRST QLORA TRAINING STEP: SUCCESS")
