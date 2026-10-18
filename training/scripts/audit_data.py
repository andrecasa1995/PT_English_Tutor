"""Read-only training/evaluation dataset leakage and schema audit."""
import json,unicodedata
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'datasets'
sets={name:json.loads((root/name).read_text(encoding='utf-8'))['examples'] for name in ('seed_examples.json','heldout_eval.json')}
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
all_ids=set();all_groups=set();all_texts=set(); problems=[]
required={'original','replacement','category','explanation_pt'}
for name, rows in sets.items():
 for row in rows:
  for key,val,seen in [('id',row['id'],all_ids),('group',row['group'],all_groups),('text',norm(row['text']),all_texts)]:
   if val in seen:problems.append(f'{name}/{row["id"]}: duplicate {key}')
   seen.add(val)
  for edit in row['edits']:
   if set(edit)!=required: problems.append(f'{name}/{row["id"]}: wrong edit fields')
   if edit['original'].strip()!=edit['original']: problems.append(f'{name}/{row["id"]}: padded original span')
   if any(ord(c)<32 for c in edit['replacement']):problems.append(f'{name}/{row["id"]}: control character')
print(f'[step 128] Train={len(sets["seed_examples.json"])}; holdout={len(sets["heldout_eval.json"])}')
print(f'[step 128] Duplicate/schema findings={len(problems)}')
for p in problems:print('[step 128] '+p)
if problems:raise SystemExit(1)
print('[step 128] DATASET LEAKAGE AUDIT: PASS')
