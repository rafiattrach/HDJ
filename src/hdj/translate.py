"""Offline DE↔EN translation using argostranslate."""

import logging
import re

logger = logging.getLogger(__name__)

# Lazy-loaded translation objects (set on first use)
_de_en = None
_en_de = None

# Common German function words used for language detection
_GERMAN_MARKERS = frozenset({
    "der", "die", "das", "und", "ist", "für", "mit", "auf", "den", "dem",
    "ein", "eine", "des", "von", "zu", "nicht", "sich", "auch", "werden",
    "als", "wird", "sind", "aus", "dass", "oder", "nach", "über", "bei",
    "wie", "nur", "noch", "aber", "ihre", "wenn", "kann", "dieser", "diese",
    "einem", "eines", "einer", "keine", "durch", "können", "haben", "mehr",
    "sehr", "müssen", "bereits", "zwischen", "sowie", "jedoch", "sollen",
    "wir", "alle", "zum", "zur", "gegen", "vom", "vor", "hier", "bis",
    "unter", "während", "sollte", "ohne", "wurde", "werden",
    "im", "ans", "beim", "aufs", "ums",  # contractions
})

# Regex for German-specific characters
_UMLAUT_RE = re.compile(r"[äöüÄÖÜß]")


def ensure_packages() -> None:
    """Download DE↔EN language packs if not already installed (one-time)."""
    import argostranslate.package

    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    installed_langs = {
        (p.from_code, p.to_code)
        for p in argostranslate.package.get_installed_packages()
    }

    for from_code, to_code in [("de", "en"), ("en", "de")]:
        if (from_code, to_code) in installed_langs:
            continue
        pkg = next(
            (
                p
                for p in available
                if p.from_code == from_code and p.to_code == to_code
            ),
            None,
        )
        if pkg is None:
            logger.warning("Language pack %s→%s not found in index", from_code, to_code)
            continue
        logger.info("Installing language pack %s→%s …", from_code, to_code)
        pkg.install()


def _get_translator(from_code: str, to_code: str):
    """Return a cached argostranslate translator for the given direction."""
    import argostranslate.translate

    languages = {
        lang.code: lang for lang in argostranslate.translate.get_installed_languages()
    }
    from_lang = languages.get(from_code)
    to_lang = languages.get(to_code)
    if from_lang is None or to_lang is None:
        raise RuntimeError(
            f"Language pack {from_code}→{to_code} not installed. "
            "Run ensure_packages() first."
        )
    translator = from_lang.get_translation(to_lang)
    if translator is None:
        raise RuntimeError(
            f"No translation available for {from_code}→{to_code}."
        )
    return translator


def _load_translators():
    """Lazy-load both translation directions, installing packages if needed."""
    global _de_en, _en_de
    if _de_en is None or _en_de is None:
        ensure_packages()
    if _de_en is None:
        _de_en = _get_translator("de", "en")
    if _en_de is None:
        _en_de = _get_translator("en", "de")


def detect_language(text: str) -> str:
    """Heuristic language detection: returns ``"de"`` or ``"en"``.

    Checks for German-specific characters (ä/ö/ü/ß) and common German
    function words.  Defaults to ``"en"`` when uncertain.
    """
    if not text or not text.strip():
        return "en"

    # Quick check: German-specific characters
    if _UMLAUT_RE.search(text):
        return "de"

    # Word-level check: proportion of German function words
    words = set(re.findall(r"\w+", text.lower()))
    if not words:
        return "en"

    german_hits = words & _GERMAN_MARKERS
    if len(german_hits) >= 2 or (len(german_hits) == 1 and len(words) <= 5):
        return "de"

    return "en"


def translate_to_english(text: str) -> str:
    """Translate *text* to English.  Returns original if already English."""
    if not text or not text.strip():
        return text
    if detect_language(text) == "en":
        return text
    try:
        _load_translators()
        return _de_en.translate(text)
    except Exception:
        logger.warning("DE→EN translation failed, returning original", exc_info=True)
        return text


def translate_to_german(text: str) -> str:
    """Translate *text* to German.  Returns original if already German."""
    if not text or not text.strip():
        return text
    if detect_language(text) == "de":
        return text
    try:
        _load_translators()
        return _en_de.translate(text)
    except Exception:
        logger.warning("EN→DE translation failed, returning original", exc_info=True)
        return text


def translate_query(query: str) -> tuple[str, str | None]:
    """Detect language of *query* and translate to the other language.

    Returns ``(original, translated)`` where *translated* is the query in
    the other language, or ``None`` if translation fails or is unavailable.
    """
    if not query or not query.strip():
        return (query, None)

    lang = detect_language(query)
    try:
        _load_translators()
        if lang == "de":
            translated = _de_en.translate(query)
        else:
            translated = _en_de.translate(query)
        return (query, translated)
    except Exception:
        logger.warning("Query translation failed, returning original only", exc_info=True)
        return (query, None)
