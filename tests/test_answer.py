from askdocs.answer import cited_indices
from askdocs.chunking import Chunk
from askdocs.eval.generate import validate
from askdocs.llm import parse_json

CHUNK = Chunk(id="a#0", doc_path="a.md", title="A", section="A", anchor=None,
              text="To declare an optional query parameter, set its default value to None.")


def test_cited_indices_dedupes_and_drops_out_of_range():
    assert cited_indices("Yes [2]. Also [1][2] and [9].", 5) == [1, 2]
    assert cited_indices("No citations.", 5) == []


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_validate_accepts_good_item():
    item = {"question": "How do I make a query param optional?", "answer": "Default it to None.",
            "quote": "set its default value to None."}
    assert validate(item, CHUNK) is None


def test_validate_rejects_bad_items():
    good = {"question": "How do I make a query param optional?", "answer": "Default to None.",
            "quote": "To declare an optional query parameter, set its default value to None."}
    assert validate({**good, "question": "What does the passage say about defaults?"}, CHUNK) == "question not standalone"
    assert validate({**good, "quote": "Use Query(max_length=50) to add validation rules."}, CHUNK) == "quote not found in source"
    assert validate({**good, "question": "Optional params"}, CHUNK) == "not a question"
    assert validate({"skip": True}, CHUNK) == "skipped by model"
