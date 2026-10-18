"""Basic validation tests; no Ollama connection required."""
import unittest

from app.analysis import validate_edits, merge_edits


class ValidationTests(unittest.TestCase):
    def test_locates_exact_span(self):
        text = "The modelo learns."
        edits = [{"original": "modelo", "replacement": "model",
                  "category": "translation", "explanation_pt": "Modelo em inglês."}]
        result = validate_edits(text, edits)
        self.assertEqual((result[0].start, result[0].end), (4, 10))

    def test_five_distinct_corrections_keep_original_offsets(self):
        text = "The modelo learn from dados and make previsoes based on previous examples."
        pairs = [("modelo", "model", "translation"), ("learn", "learns", "grammar"),
                 ("dados", "data", "translation"), ("make", "makes", "grammar"),
                 ("previsoes", "predictions", "translation")]
        edits = [{"original": original, "replacement": replacement,
                  "category": category, "explanation_pt": "Explicacao."}
                 for original, replacement, category in pairs]
        actual = validate_edits(text, edits)
        self.assertEqual(len(actual), 5)
        self.assertEqual([(text[x.start:x.end], x.replacement, x.category) for x in actual], pairs)

    def test_neural_network_example_validates_four_edits(self):
        text = "This neural network have multiple camadas and learn from exemplos."
        pairs = [("have", "has", "grammar"), ("camadas", "layers", "translation"),
                 ("learn", "learns", "grammar"), ("exemplos", "examples", "translation")]
        edits = [{"original": a, "replacement": b, "category": c,
                  "explanation_pt": "Teste."} for a, b, c in pairs]
        output = validate_edits(text, edits)
        self.assertEqual([(s.original, s.replacement, s.category) for s in output], pairs)

    def test_merge_rejects_wrong_categories_for_pass(self):
        text = "The modelo have layers."
        translations = [
            {"original": "modelo", "replacement": "model", "category": "translation", "explanation_pt": "Modelo."},
            {"original": "have", "replacement": "has", "category": "grammar", "explanation_pt": "Concordancia."},
        ]
        grammar = [
            {"original": "have", "replacement": "has", "category": "grammar", "explanation_pt": "Concordancia."},
            {"original": "modelo", "replacement": "model", "category": "translation", "explanation_pt": "Modelo."},
        ]
        output = merge_edits(text, translations, grammar)
        self.assertEqual([(x.original, x.category) for x in output],
                         [("modelo", "translation"), ("have", "grammar")])

    def test_translation_has_priority_on_overlapping_span(self):
        text = "The resultado is ready."
        translations = [{"original": "resultado", "replacement": "result",
                         "category": "translation", "explanation_pt": "Tradução para inglês."}]
        grammar = [{"original": "resultado", "replacement": "result",
                    "category": "spelling", "explanation_pt": "Categoria incorreta."}]
        output = merge_edits(text, translations, grammar)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].category, "translation")
        self.assertEqual(output[0].explanation_pt, "Tradução para inglês.")

    def test_rejects_nonexistent_span(self):
        edit = [{"original": "banana", "replacement": "apple",
                 "category": "grammar", "explanation_pt": "Teste."}]
        self.assertEqual(validate_edits("The model learns.", edit), [])

    def test_rejects_ambiguous_repeated_text(self):
        edit = [{"original": "data", "replacement": "dataset",
                 "category": "style", "explanation_pt": "Teste."}]
        self.assertEqual(validate_edits("data and data", edit), [])


if __name__ == "__main__":
    unittest.main()
