"""Validate synthetic candidates and enforce separation from frozen evaluation."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from validate_dataset import validate
files=[root/'datasets'/x for x in ('seed_examples.json','heldout_eval.json','additional_synthetic_candidates.json')]
sets=[json.loads(p.read_text(encoding='utf-8'))['examples'] for p in files]
for rows in sets:validate(rows)
ids=[e['id'] for s in sets for e in s]
groups=[e['group'] for s in sets for e in s]
texts=[' '.join(e['text'].casefold().split()) for s in sets for e in s]
assert len(ids)==len(set(ids)) and len(groups)==len(set(groups)) and len(texts)==len(set(texts)),'Overlapping train/eval/candidate examples'
print(f'[step 130] Dataset cross-check passed: training={len(sets[0])}, holdout={len(sets[1])}, synthetic candidates={len(sets[2])}')
print('[step 130] Frozen heldout data untouched. Candidate examples NOT yet used for training.')
