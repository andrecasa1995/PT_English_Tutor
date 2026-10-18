"""Markdown protection smoke tests."""
import unittest
from app.analysis import validate_edits
from app.markdown_guard import protected_ranges


def edit(original):
    return {"original": original, "replacement": "changed", "category": "style",
            "explanation_pt": "Teste."}


class MarkdownGuardTests(unittest.TestCase):
    def test_wikilink_and_code_are_protected(self):
        text = "Read [[Neural Networks]] then run `model.fit(X, y)`."
        self.assertEqual(validate_edits(text, [edit("Neural Networks"), edit("model.fit(X, y)")]), [])

    def test_regular_word_outside_markdown_is_allowed(self):
        text = "The modelo uses [[Neural Networks]]."
        self.assertEqual(len(validate_edits(text, [edit("modelo")])), 1)

    def test_frontmatter_is_protected(self):
        text = "---\ntitle: modelos\n---\nThe model works."
        self.assertEqual(validate_edits(text, [edit("modelos")]), [])

    def test_fenced_code_is_protected(self):
        text = "```python\nmodel.fit(dados)\n```\nThe model works."
        self.assertEqual(validate_edits(text, [edit("dados")]), [])


if __name__ == "__main__":
    unittest.main()
