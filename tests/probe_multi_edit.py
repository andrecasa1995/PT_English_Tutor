import json,time,urllib.request
text='The student have a algoritm and write the resultado.'
req=urllib.request.Request('http://127.0.0.1:8765/v1/analyze',data=json.dumps({'text':text}).encode(),headers={'Content-Type':'application/json'})
t=time.monotonic()
with urllib.request.urlopen(req,timeout=150) as r: data=json.load(r)
print('TIME:',round(time.monotonic()-t,2),flush=True)
print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)
