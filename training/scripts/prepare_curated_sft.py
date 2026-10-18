"""Export experimental SFT JSONL: 12 seeds + curated synthetic; never read holdout."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training'))
from validate_dataset import validate
from scripts.prepare_sft import convert
seed=json.loads((ROOT/'training/datasets/seed_examples.json').read_text(encoding='utf-8'))['examples']
curated=json.loads((ROOT/'training/datasets/curated_synthetic_v1.json').read_text(encoding='utf-8'))['examples']
heldout=json.loads((ROOT/'training/datasets/heldout_eval.json').read_text(encoding='utf-8'))['examples']
train=seed+curated
validate(train)
texts=lambda a:{' '.join(x['text'].casefold().split()) for x in a}
assert not (texts(train)&texts(heldout)), 'Training/evaluation leakage'
assert not ({r['group'] for r in train}&{r['group'] for r in heldout}), 'Group leakage'
assert len(train)==49
out=ROOT/'training/data/sft_curated_v1.jsonl'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8') as file:
 for row in train:file.write(json.dumps(convert(row),ensure_ascii=False)+'\n')
print(f'[step 134] Exported {len(train)} experimental supervised examples; heldout set excluded: {out}')
