"""HTTP contract tests without a running Ollama service."""
import unittest
from unittest.mock import AsyncMock, patch
import httpx
from fastapi.testclient import TestClient
from app.main import app

class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app)

    def test_health(self):
        result=self.client.get('/health')
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json()['status'],'ok')
        self.assertTrue(result.json()['model'])

    def test_empty_text_rejected(self):
        for endpoint in ('/v1/correct','/v1/analyze'):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.client.post(endpoint,json={'text':''}).status_code,422)

    def test_oversized_text_rejected(self):
        for endpoint in ('/v1/correct','/v1/analyze'):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.client.post(endpoint,json={'text':'A'*2001}).status_code,422)

    def test_correct_requires_approval(self):
        response=httpx.Response(200,json={'response':'She goes to class.'},request=httpx.Request('POST','http://127.0.0.1:11434/api/generate'))
        with patch('httpx.AsyncClient.post',new_callable=AsyncMock,return_value=response) as mocked:
            got=self.client.post('/v1/correct',json={'text':'She go to class.'})
        self.assertEqual(got.status_code,200)
        self.assertEqual(got.json()['suggestion'],'She goes to class.')
        self.assertEqual(got.json()['original'],'She go to class.')
        self.assertIs(got.json()['requires_approval'],True)
        self.assertFalse(mocked.call_args.kwargs['json']['stream'])

    def test_correct_ollama_unavailable(self):
        with patch('httpx.AsyncClient.post',new_callable=AsyncMock,side_effect=httpx.ConnectError('offline')):
            got=self.client.post('/v1/correct',json={'text':'Hello world.'})
        self.assertEqual(got.status_code,503)
        self.assertNotIn('offline',got.text)

if __name__=='__main__': unittest.main()
