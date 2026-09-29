import os
import yaml
import requests
import logging
import time
from pathlib import Path

from language.guardrail import check_hallucination, fallback_template_join

logger = logging.getLogger("samvaad.language")

class GlossToTextEngine:
    def __init__(self, backend="ollama", model_name="llama3.2:3b", examples_path="data/gloss_examples.yaml"):
        self.backend = backend
        self.model_name = model_name
        
        # Load few-shot examples
        self.examples = []
        try:
            with open(examples_path, 'r') as f:
                data = yaml.safe_load(f)
                self.examples = data.get('examples', [])
        except Exception as e:
            logger.warning(f"Could not load few-shot examples from {examples_path}: {e}")
            
        self.system_prompt = (
            "You are an expert Indian Sign Language (ISL) translator. "
            "Your task is to rewrite a sequence of ISL glosses into exactly ONE short, natural English sentence. "
            "STRICT RULES: Use ONLY the meaning present in the glosses. DO NOT invent facts, names, times, or places. "
            "If the meaning is unclear, just output the glosses joined together."
        )
        
    def _build_prompt(self, gloss_sequence: str) -> str:
        prompt = self.system_prompt + "\n\n"
        
        # Add few-shot examples
        for ex in self.examples:
            prompt += f"Gloss: {ex['gloss']}\nEnglish: {ex['english']}\n\n"
            
        prompt += f"Gloss: {gloss_sequence}\nEnglish:"
        return prompt

    def _generate_ollama(self, prompt: str) -> str:
        """Hits the local Ollama API for generation."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1, # Keep it extremely deterministic
                        "num_predict": 30
                    }
                },
                timeout=2.0 # Fast timeout, if LLM is slow, we must fallback
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama backend failed: {e}")
            return ""

    def translate(self, gloss_sequence: str) -> dict:
        """
        Translates gloss to text using LLM, verified by Guardrail.
        Falls back to rule-based joining on failure or hallucination.
        """
        if not gloss_sequence:
            return {"text": "", "backend": "none", "latency_ms": 0}
            
        start_time = time.time()
        
        # 1. Generate candidate translation
        candidate = ""
        used_backend = self.backend
        
        if self.backend == "ollama":
            prompt = self._build_prompt(gloss_sequence)
            candidate = self._generate_ollama(prompt)
            
        # 2. Guardrail Check
        if not candidate or check_hallucination(gloss_sequence, candidate):
            # Fallback triggered!
            candidate = fallback_template_join(gloss_sequence)
            used_backend = "rule_fallback"
            
        latency = int((time.time() - start_time) * 1000)
        
        return {
            "text": candidate,
            "backend": used_backend,
            "latency_ms": latency
        }
