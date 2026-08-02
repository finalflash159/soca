from __future__ import annotations

from eval.tts_intelligibility.corpora import (
    CorpusItem,
    build_all_corpora,
    control_corpus,
    lexicon_corpus,
    normalizer_corpus,
)


class TestNormalizerCorpus:
    def test_is_not_empty(self) -> None:
        assert len(normalizer_corpus()) > 0

    def test_expects_the_spoken_form_not_the_written_form(self) -> None:
        items = {item.text_in: item for item in normalizer_corpus()}
        item = items["Giá là 100.000đ"]
        assert item.expected == "một trăm nghìn đồng"
        assert item.mode == "contains"

    def test_every_item_has_a_non_empty_expectation(self) -> None:
        assert all(item.expected.strip() for item in normalizer_corpus())

    def test_item_ids_are_unique(self) -> None:
        ids = [item.item_id for item in normalizer_corpus()]
        assert len(ids) == len(set(ids))


class TestLexiconCorpus:
    def test_draws_from_the_curated_valtec_lexicon(self) -> None:
        from soca.tts.valtec.lexicon import WORD_LEXICON

        terms = {item.expected for item in lexicon_corpus()}
        assert terms & set(WORD_LEXICON)

    def test_wraps_each_term_in_a_carrier_sentence(self) -> None:
        # A bare term gives the TTS no prosodic context and gives the ASR no
        # language-model context, so both sides behave unlike real use.
        for item in lexicon_corpus():
            assert item.expected in item.text_in
            assert len(item.text_in) > len(item.expected)

    def test_uses_term_mode(self) -> None:
        assert all(item.mode == "term" for item in lexicon_corpus())

    def test_limit_caps_the_corpus_size(self) -> None:
        assert len(lexicon_corpus(limit=5)) == 5

    def test_is_deterministic_across_calls(self) -> None:
        assert [i.item_id for i in lexicon_corpus(limit=10)] == [
            i.item_id for i in lexicon_corpus(limit=10)
        ]


class TestControlCorpus:
    def test_is_plain_vietnamese_without_technical_terms(self) -> None:
        assert len(control_corpus()) > 0
        for item in control_corpus():
            assert item.mode == "exact"
            assert item.text_in.isascii() is False  # Vietnamese diacritics present

    def test_expected_is_the_spoken_form_of_the_input(self) -> None:
        for item in control_corpus():
            assert item.expected.strip()


class TestBuildAllCorpora:
    def test_returns_every_corpus_keyed_by_name(self) -> None:
        corpora = build_all_corpora(lexicon_limit=5)
        assert set(corpora) == {"normalizer", "lexicon", "control"}

    def test_item_ids_are_globally_unique(self) -> None:
        corpora = build_all_corpora(lexicon_limit=5)
        ids = [item.item_id for items in corpora.values() for item in items]
        assert len(ids) == len(set(ids))

    def test_items_are_immutable(self) -> None:
        import dataclasses

        import pytest

        item = normalizer_corpus()[0]
        assert isinstance(item, CorpusItem)
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.expected = "changed"  # type: ignore[misc]
