"""One-shot, unattended, bounded local QA workflow. Never trains, deploys, pushes, or edits the holdout."""
import datetime,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'training/scripts'
LOG=ROOT/'training/training_live.log'
STATUS=ROOT/'training/outputs/workflow_status.json'
STATUS.parent.mkdir(parents=True,exist_ok=True)
STAGES=[('dataset_validation',ROOT/'training/validate_dataset.py'),('candidate_validation',SCRIPTS/'validate_candidates.py'),('dataset_audit',SCRIPTS/'audit_data.py'),('checkpoint_validation',SCRIPTS/'verify_curated_adapter.py')]
def emit(message):
 line='[workflow] '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+message
 print(line,flush=True)
 with LOG.open('a',encoding='utf-8') as f:f.write(line+'\n')
def save(state):
 temp=STATUS.with_suffix('.tmp')
 temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
 temp.replace(STATUS)
def main():
 state={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','stages':[]}
 save(state);emit('Starting bounded validation workflow')
 for name,script in STAGES:
  emit('Running '+name)
  cp=subprocess.run([sys.executable,str(script)],cwd=str(ROOT),text=True,capture_output=True,timeout=180)
  with LOG.open('a',encoding='utf-8') as f:f.write(cp.stdout+cp.stderr)
  entry={'name':name,'exit_code':cp.returncode,'output_tail':(cp.stdout+cp.stderr)[-1500:]}
  state['stages'].append(entry);save(state)
  if cp.returncode:
   state['status']='failed';save(state);emit('FAILED '+name);return 1
  emit('PASS '+name)
 state['status']='complete';state['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(state)
 emit('WORKFLOW COMPLETE: all validations passed')
 return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:emit('FATAL '+str(exc));traceback.print_exc();sys.exit(1)
