import unittest
from app.analysis import validate_edits

class ExplanationValidationTests(unittest.TestCase):
    def test_rejects_empty_explanation(self):
        sample='She go to school.'
        edits=[{'original':'go','replacement':'goes','category':'grammar','explanation_pt':'   '}]
        self.assertEqual(validate_edits(sample,edits),[])

    def test_preserves_valid_portuguese_explanation(self):
        sample='She go to school.'
        edits=[{'original':'go','replacement':'goes','category':'grammar','explanation_pt':'Com she, o verbo recebe -es.'}]
        result=validate_edits(sample,edits)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].explanation_pt,edits[0]['explanation_pt'])
