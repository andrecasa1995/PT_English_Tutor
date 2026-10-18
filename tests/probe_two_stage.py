import json,urllib.request,time
cases=[('word','algoritm'),('word','resultado'),('sentence','The student have a algoritm and write the resultado.')]
for mode,text in cases:
 req=urllib.request.Request('http://127.0.0.1:8765/v1/analyze',data=json.dumps({'text':text,'mode':mode}).encode(),headers={'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=100) as response: data=json.load(response)
 print(mode,round(time.monotonic()-start,1),[(s['original'],s['replacement'],s['category']) for s in data['suggestions']],flush=True)
 assert data['requires_approval'] is True
 assert all(text[s['start']:s['end']]==s['original'] for s in data['suggestions'])
