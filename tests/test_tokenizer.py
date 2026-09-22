import pytest

from plastic.tokenizer.bpe import Tokenizer

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs! Sphinx of black quartz, judge my vow.\n"
    "Numbers 12345 and symbols #$% and tabs\there and unicode café naïve façade.\n"
) * 20


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train(SAMPLE.splitlines(), vocab_size=300)


def test_vocab_size_and_special_ids(tok):
    assert tok.vocab_size == 300
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2


def test_exact_roundtrip_with_whitespace(tok):
    text = "hello  world\tfoo\nbar   baz\n\n  indented café"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bos_eos_and_skip_special(tok):
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "hello"
    assert tok.decode(ids, skip_special=False) != "hello"


def test_different_words_differ(tok):
    assert tok.encode("fox") != tok.encode("dog")


def test_save_load_identical(tok, tmp_path):
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    back = Tokenizer.load(path)
    text = "Pack my box with five dozen liquor jugs!"
    assert back.encode(text) == tok.encode(text)
    assert back.vocab_size == tok.vocab_size
