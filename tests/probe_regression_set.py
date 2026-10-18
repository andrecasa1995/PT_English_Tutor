"""Small exploratory live Ollama regression probe; prints observations, no training."""
import json,time,urllib.request
cases=[('correct','The student writes the result correctly.'),('multi','The student have a algoritm and write the resultado.'),('markdown','The student have an error in `write_code()` and the [documentation](https://example.com) is available.')]
for label,sentence in cases:
 payload=json.dumps({'text':sentence}).encode('utf-8')
 req=urllib.request.Request('http://127.0.0.1:8765/v1/analyze',data=payload,headers={'Content-Type':'application/json'})
 started=time.monotonic()
 try:
  with urllib.request.urlopen(req,timeout=150) as response:data=json.load(response)
  edits=data.get('suggestions',[])
  assert all(sentence[e['start']:e['end']]==e['original'] for e in edits)
  print(f'[regression] {label}: {len(edits)} edits in {time.monotonic()-started:.1f}s',flush=True)
  for e in edits:print(f'  {e["original"]!r} -> {e["replacement"]!r} ({e["category"]})',flush=True)
 except Exception as exc:print(f'[regression] {label} FAILED: {exc}',flush=True);raise
