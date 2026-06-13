"""Tests for the bibliography / citation detector used to filter search noise."""

from hdj.rag import is_reference_like


class TestIsReferenceLike:
    def test_bibliography_block_is_flagged(self):
        text = (
            "Taylor, L. (2017) What is data justice? Big Data & Society, 4(2), pp. 1-14. "
            "doi:10.1177/2053951717736335. "
            "Johnson, J. A. (2014) From open data to information justice. Ethics and "
            "Information Technology, 16(4), pp. 263-274. "
            "Yadav, A. (2016) Aadhaar and the poor. Retrieved from https://example.org/aadhaar."
        )
        assert is_reference_like(text) is True

    def test_prose_with_one_citation_is_kept(self):
        text = (
            "Aadhaar is what Johnson (2014) terms a 'disciplinary system'. It raises "
            "several issues of justice that are specific to its use of data technologies, "
            "particularly in the way it records, stores and processes biometric data about "
            "the poorest and most marginalised populations who cannot opt out of enrolment."
        )
        assert is_reference_like(text) is False

    def test_plain_prose_is_kept(self):
        text = (
            "Data justice means fairness in the way people are made visible, represented "
            "and treated as a result of their production of digital data. It concerns the "
            "distribution of both the benefits and the harms of data technologies."
        )
        assert is_reference_like(text) is False

    def test_short_text_never_flagged(self):
        assert is_reference_like("(2016)") is False
        assert is_reference_like("") is False

    def test_url_heavy_footnote_is_flagged(self):
        text = (
            "See https://example.org/a and https://example.org/b and https://example.org/c "
            "and https://example.org/d for the underlying datasets and methodology notes."
        )
        assert is_reference_like(text) is True

    def test_prose_with_several_inline_citations_is_kept(self):
        # Argument prose often cites several sources in one or two sentences;
        # parenthetical years alone must not get it dropped from results.
        text = (
            "The Aadhaar program (2016) expanded biometric enrollment. As Taylor "
            "(2017) and Johnson (2014) argue, data justice requires fairness in how "
            "such systems treat the poorest and most marginalised populations."
        )
        assert is_reference_like(text) is False

    def test_year_only_bibliography_is_flagged(self):
        # A reference list made purely of "(year)" entries, with no DOIs/URLs.
        text = (
            "Smith, J. (2010). Jones, K. (2011). Brown, L. (2012). "
            "Lee, M. (2013). Park, S. (2014). Kim, T. (2015)."
        )
        assert is_reference_like(text) is True
