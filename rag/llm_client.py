"""Gemini client for the generative step of the RAG pipeline."""

from __future__ import annotations

import os

# Enables direct use of this module outside the main pipeline as well.
import config  # noqa: F401


class LLMNotConfiguredError(RuntimeError):
    """Raised when generation was requested without an LLM configuration."""


def is_configured() -> bool:
    """Whether this process has the credentials required for Gemini generation."""
    return bool(os.getenv("GEMINI_API_KEY"))


def generate_completion(prompt: str, instructions: str) -> str:
    """Generate text with the Gemini Interactions API.

    The API key is read only from ``GEMINI_API_KEY``; it must never be placed
    in source code or committed configuration.  ``RAG_LLM_MODEL`` is optional
    so deployments can choose the model appropriate to their cost and quality
    requirements without changing the application.
    """
    if not is_configured():
        raise LLMNotConfiguredError("GEMINI_API_KEY is not configured.")
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError("Install the 'google-genai' package to enable Gemini answer generation.") from error

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    interaction = client.interactions.create(
        model=os.getenv("RAG_LLM_MODEL", "gemini-3.8-flash"),
        system_instruction=instructions,
        input=prompt,
    )
    answer = interaction.output_text.strip()
    if not answer:
        raise RuntimeError("The LLM returned no text.")
    return answer
