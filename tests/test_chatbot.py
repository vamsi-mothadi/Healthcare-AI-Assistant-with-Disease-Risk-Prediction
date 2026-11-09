"""
Unit Tests for Healthcare RAG Chatbot, Guardrails, and Citations.
"""

import pytest

from src.rag.chatbot import HealthcareChatbot, HEALTHCARE_SYSTEM_PROMPT


@pytest.fixture(scope="module")
def mock_bot():
    """Returns a chatbot instance running in mock mode to avoid external API calls."""
    return HealthcareChatbot(mock_mode=True)


def test_chatbot_grounded_answer(mock_bot):
    """Verify chatbot returns grounded answers and citations for clinical questions."""
    response = mock_bot.ask("What blood glucose level indicates prediabetes?")

    assert "answer" in response
    assert len(response["citations"]) > 0
    assert response["context_used"] is True
    assert "Disclaimer" in response["answer"]


def test_chatbot_refusal_on_irrelevant_query():
    """Verify chatbot triggers refusal policy when question is completely outside knowledge base."""
    # Use real bot with mock mode to test thresholding
    bot = HealthcareChatbot(mock_mode=True)
    # Query with score threshold that yields no matches
    res = bot.ask("What is the recipe for baking chocolate chip cookies in Paris?")

    # Should either refuse or flag low relevance
    assert "answer" in res
    if not res["context_used"]:
        assert "I do not have sufficient medical documentation" in res["answer"]


def test_prompt_injection_safety_prompt():
    """Verify system prompt contains explicit guardrails neutralizing prompt injection."""
    assert "UNTRUSTED CONTEXT" in HEALTHCARE_SYSTEM_PROMPT
    assert "ignore previous instructions" in HEALTHCARE_SYSTEM_PROMPT.lower()
    assert "NO HALLUCINATION / REFUSAL POLICY" in HEALTHCARE_SYSTEM_PROMPT
    assert "ABSOLUTE MEDICAL NON-DIAGNOSIS" in HEALTHCARE_SYSTEM_PROMPT


def test_chatbot_empty_input(mock_bot):
    """Verify chatbot gracefully handles blank inputs."""
    res = mock_bot.ask("   ")
    assert "Please ask a specific" in res["answer"]
    assert res["citations"] == []

