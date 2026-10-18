"""Generate additional synthetic training-only examples; never read held-out labels."""
import json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
src=r/'datasets/seed_examples.json'
out=r/'datasets/additional_synthetic_candidates.json'
existing=json.loads(src.read_text(encoding='utf-8'))['examples']
cases=[
('grammar','She use the application every day.','use','uses','Com she, o verbo no presente simples recebe -s.'),
('grammar','The program generate a new report.','generate','generates','Program é singular: generates.'),
('grammar','My teammate work remotely.','work','works','Teammate está no singular: works.'),
('grammar','The API return an error.','return','returns','O sujeito API é singular: returns.'),
('grammar','Every student need feedback.','need','needs','Every student é singular: needs.'),
('grammar','These functions runs quickly.','runs','run','Functions está no plural: run.'),
('grammar','Our developers writes unit tests.','writes','write','Developers está no plural: write.'),
('grammar','Two server are online.','server','servers','Two exige o plural servers.'),
('grammar','Several model were evaluated.','model','models','Several exige o substantivo plural.'),
('grammar','Those result look promising.','result','results','Those exige o substantivo no plural: results.'),
('spelling','Please chek the configuration.','chek','check','A grafia correta é check.'),
('spelling','The respose was delayed.','respose','response','A grafia correta é response.'),
('spelling','I updated the dependecies.','dependecies','dependencies','A grafia correta é dependencies.'),
('spelling','We should optimze the query.','optimze','optimize','A grafia correta é optimize.'),
('spelling','The model uses paralel processing.','paralel','parallel','A grafia correta é parallel.'),
('spelling','The output is inconsistant.','inconsistant','inconsistent','A grafia correta é inconsistent.'),
('spelling','He reviewed the algoritm.','algoritm','algorithm','A grafia correta é algorithm.'),
('spelling','The functon accepts two arguments.','functon','function','A grafia correta é function.'),
('translation','I need to revisar the documentation.','revisar','review','Nesse contexto, revisar corresponde a review.'),
('translation','We fixed the erro in production.','erro','error','Erro corresponde a error nesse contexto.'),
('translation','The equipe completed the deployment.','equipe','team','Equipe corresponde a team.'),
('translation','Please update the arquivo today.','arquivo','file','Arquivo corresponde a file.'),
('translation','The servidor is offline.','servidor','server','Servidor corresponde a server nesse contexto.'),
('translation','My colega tested the feature.','colega','colleague','Colega corresponde a colleague.'),
('translation','I submitted the tarefa yesterday.','tarefa','assignment','Tarefa corresponde a assignment nesse contexto acadêmico.'),
('translation','The reunião starts at nine.','reunião','meeting','Reunião corresponde a meeting.'),
]
clean=[
'Our deployment passed all checks.',
'The model runs locally on my laptop.',
'My colleagues reviewed the pull request.',
'We use Git to track source code.',
'This API supports multiple clients.',
'The training dataset contains synthetic examples.',
'I submitted my assignment before the deadline.',
'The application starts without errors.',
'These results are consistent with the tests.',
'The documentation explains the setup clearly.',
'My professor gave us feedback.',
'The system stores its configuration locally.',
]
rows=[]
for i,(cat,text,original,replacement,explanation) in enumerate(cases,1):
 rows.append({'id':f'candidate-{i:03d}','group':f'candidate-{i:03d}','text':text,'edits':[{'original':original,'replacement':replacement,'category':cat,'explanation_pt':explanation}]})
for i,text in enumerate(clean,1):
 rows.append({'id':f'candidate-clean-{i:03d}','group':f'candidate-clean-{i:03d}','text':text,'edits':[]})
assert len(rows)==38
assert not out.exists(),'Do not overwrite candidates'
out.write_text(json.dumps({'schema_version':'0.1','status':'synthetic_candidates_pending_review','examples':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'[step 129] Wrote {len(rows)} SYNTHETIC CANDIDATES (not added to training yet): {out}')
