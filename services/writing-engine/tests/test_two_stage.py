"""Verifies that word and sentence modes call only their intended analysis passes."""
import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app

class TwoStageTests(unittest.TestCase):
    def test_word_uses_one_combined_translation_and_spelling_pass(self):
        with patch('app.analysis._analyze_pass', new_callable=AsyncMock, return_value=[]) as call:
            response = TestClient(app).post('/v1/analyze', json={'text':'algoritm', 'mode':'word'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(call.await_count, 1)
        self.assertIn('translation', call.await_args.args[2].lower())
        self.assertIn('spelling', call.await_args.args[2].lower())

    def test_sentence_only_uses_grammar(self):
        with patch('app.analysis._analyze_pass', new_callable=AsyncMock, return_value=[]) as call:
            response = TestClient(app).post('/v1/analyze', json={'text':'She go.', 'mode':'sentence'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(call.await_count, 1)
        self.assertIn('proofreader', call.await_args.args[2].lower())

    def test_invalid_mode_rejected(self):
        self.assertEqual(TestClient(app).post('/v1/analyze', json={'text':'Hello', 'mode':'invalid'}).status_code, 422)
