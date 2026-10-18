"""Curate conservative synthetic examples; keep holdout untouched."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'datasets/additional_synthetic_candidates.json'
dst=ROOT/'datasets/curated_synthetic_v1.json'
rows=json.loads(src.read_text(encoding='utf-8'))['examples']
excluded={'candidate-025': 'tarefa -> assignment needs explicit academic context'}
selected=[row for row in rows if row['id'] not in excluded]
assert len(selected)==37 and len(rows)==38
assert not dst.exists(), 'Refusing to replace curated dataset'
out={'schema_version':'0.1','status':'synthetic_curated_for_experimental_training_only','excluded':excluded,'examples':selected}
dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'[step 133] Curated {len(selected)} experimental training candidates; excluded {len(excluded)} ambiguous example')
