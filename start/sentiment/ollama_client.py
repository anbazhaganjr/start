"""
Ollama REST API client for local LLM sentiment analysis.

Connects to a local Ollama instance running Mistral 7B.
"""

import json
import requests
from typing import Optional

from start.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Client for Ollama REST API."""

    def __init__(
        self,
        model: str = "mistral:7b-instruct-v0.3-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model (or base name) is available
            base_name = self.model.split(":")[0]
            return any(base_name in m for m in models)
        except (requests.ConnectionError, requests.Timeout):
            return False

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt text.
            system: Optional system prompt.

        Returns:
            Model response text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for consistent sentiment
                "num_predict": 50,   # Short responses only
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.RequestException as e:
            logger.warning(f"[ollama] Request failed: {e}")
            return ""

    def analyze_sentiment(self, headline: str) -> dict:
        """
        Analyze sentiment of a single headline.

        Returns:
            Dict with 'sentiment' (-1, 0, 1) and 'confidence' (0-1).
        """
        system = (
            "You are a financial sentiment analyzer. "
            "Respond with ONLY a JSON object with two fields: "
            "'sentiment' (1 for positive, 0 for neutral, -1 for negative) "
            "and 'confidence' (0.0 to 1.0). No other text."
        )
        prompt = f"Analyze the financial sentiment of this headline:\n\"{headline}\""

        response = self.generate(prompt, system=system)

        # Parse response
        try:
            # Try to extract JSON from response
            # Handle cases where model adds extra text
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}") + 1]
                result = json.loads(json_str)
                sentiment = int(result.get("sentiment", 0))
                confidence = float(result.get("confidence", 0.5))
                # Clamp values
                sentiment = max(-1, min(1, sentiment))
                confidence = max(0.0, min(1.0, confidence))
                return {"sentiment": sentiment, "confidence": confidence}
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: keyword matching
        lower = response.lower()
        if any(w in lower for w in ["positive", "bullish", "optimistic"]):
            return {"sentiment": 1, "confidence": 0.5}
        elif any(w in lower for w in ["negative", "bearish", "pessimistic"]):
            return {"sentiment": -1, "confidence": 0.5}

        return {"sentiment": 0, "confidence": 0.3}

    def batch_analyze(self, headlines: list[str]) -> list[dict]:
        """Analyze sentiment for a batch of headlines."""
        results = []
        for headline in headlines:
            result = self.analyze_sentiment(headline)
            result["headline"] = headline
            results.append(result)
        return results
