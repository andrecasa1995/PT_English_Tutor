"""Manual local end-to-end API probe; writes no note content."""
import json,time,urllib.request,urllib.error
url='http://127.0.0.1:8765/v1/analyze'
sentence='She go to school every day.'
body=json.dumps({'text':sentence}).encode('utf-8')
start=time.monotonic()
try:
 request=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(request,timeout=150) as response:
  data=json.load(response)
 print('HTTP 200; seconds:',round(time.monotonic()-start,2),flush=True)
 print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)
 assert data['original']==sentence
 assert data['requires_approval'] is True
 for item in data['suggestions']:
  assert sentence[item['start']:item['end']]==item['original']
 print('OBSIDIAN API INTEGRATION CONTRACT: PASS',flush=True)
except Exception as error:
 print('OBSIDIAN API PROBE FAILED:',type(error).__name__,str(error),flush=True)
 raise
