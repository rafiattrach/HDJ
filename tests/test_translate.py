"""Tests for the translation / language detection module."""

import pytest

from hdj.translate import detect_language, translate_query


# ---------------------------------------------------------------------------
# detect_language — heuristic DE/EN detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_german_with_umlauts(self):
        assert detect_language("Gesundheitsdaten für alle Bürger") == "de"

    def test_german_without_umlauts(self):
        """German function words alone (no umlauts) should still be detected."""
        assert detect_language("die Daten werden durch das System verarbeitet") == "de"

    def test_english(self):
        assert detect_language("health data governance framework") == "en"

    def test_short_german_text(self):
        assert detect_language("über die Daten") == "de"

    def test_short_english_text(self):
        assert detect_language("health data") == "en"

    def test_mixed_language_with_umlauts(self):
        """Presence of umlauts should tip detection to German."""
        assert detect_language("Health data Gesundheitsdatenschutzgesetz über") == "de"

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_whitespace_only(self):
        assert detect_language("   ") == "en"

    def test_single_german_marker_short_text(self):
        """A single German marker in a short text should detect as German."""
        assert detect_language("der Algorithmus") == "de"

    def test_single_german_marker_long_text_defaults_en(self):
        """A single German marker among many English words → English."""
        assert detect_language("the algorithm processes data and generates results der") == "en"


# ---------------------------------------------------------------------------
# translate_query — round-trip translation (requires argostranslate installed)
# ---------------------------------------------------------------------------

class TestTranslation:
    """These tests require argostranslate with DE↔EN packs installed.

    They are skipped automatically if the packages are not available.
    """

    @pytest.fixture(autouse=True)
    def _check_translation_available(self):
        try:
            from hdj.translate import _load_translators
            _load_translators()
        except Exception:
            pytest.skip("argostranslate DE↔EN packs not installed")

    def test_de_to_en(self):
        original, translated = translate_query("Daten im Gesundheitswesen")
        assert original == "Daten im Gesundheitswesen"
        assert translated is not None
        assert translated != original
        # The translation should be English
        lower = translated.lower()
        assert any(w in lower for w in ("health", "data", "care", "medical"))

    def test_en_to_de(self):
        original, translated = translate_query("health data governance")
        assert original == "health data governance"
        assert translated is not None
        assert translated != original

    def test_round_trip_sanity(self):
        """Translating DE→EN→DE should produce something non-empty."""
        _, en_text = translate_query("Datenschutz im Gesundheitswesen")
        assert en_text is not None
        _, de_text = translate_query(en_text)
        assert de_text is not None
        assert len(de_text) > 0

    def test_empty_input(self):
        original, translated = translate_query("")
        assert original == ""
        assert translated is None

    def test_whitespace_input(self):
        original, translated = translate_query("   ")
        assert original == "   "
        assert translated is None
