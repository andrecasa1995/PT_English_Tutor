"""Live API long-input sanity check (no file content sent)."""
import json,time,urllib.request,urllib.error
url='http://127.0.0.1:8765/v1/analyze'

def post(text):
 request=urllib.request.Request(url,data=json.dumps({'text':text}).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(request,timeout=150) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as exc:return exc.code,None

text=('The model runs locally, and the documentation is clear. '*25)+'She go to class.'
assert len(text)<2000
start=time.monotonic();status,result=post(text)
print(f'[long] chars={len(text)}, status={status}, seconds={time.monotonic()-start:.2f}',flush=True)
assert status==200 and result['requires_approval'] is True
for edit in result['suggestions']:
 assert text[edit['start']:edit['end']]==edit['original']
print(f'[long] suggestions={len(result["suggestions"])}',flush=True)
status,_=post('X'*2001)
assert status==422
print('[long] LIMIT AND OFFSET CONTRACT: PASS',flush=True)
