"""
LLM integratoin for RAD answer generation

This module provides a clean interface for generating answers using varios LLM providers
"""

from dataclasses import dataclass
from typing import Protocol

class LLMProvider(Protocol):
    """Protocal for LLM providers"""

    def generate(self, prompt: str) -> str: ...


@dataclass
class RAGResponse:
    """Structured response from RAG system"""
    answer: str
    source: list[dict]
    prompt: str
    model: str


class OpenAIProvider:
    """OpenAI API integration"""

    def __init__(
            self,
            model: str = "gpt-4-turbo-preview",
            temperature: float = 0.1,
            max_tokens: int = 1000
    ):
        """
        Initialize OpenAI provider

        Args:
        model: OpenAI model to use
        temperature: Loewr = more deterministic (good for factual RAG)
        max_tokens: Maximum response length
        """
        from openai import OpenAI

        self.client = OpenAI() # Uses OPENAI_API_KEY env var
        self.model = model
        self.temerature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate response from OopenAI"""
        response = self.client.chat.completions.create(
            model = self.model,
            message=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about incident reviews. be concise and cite your sources"
                },
                {
                    "role": "user", "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content
    

class AnthropicProvider:
    """Anthropic Claude API integration"""

    def __init__(
            self,
            model: str = "claude-3-sonnet-20240229",
            temperature: float = 0.1,
            max_tokens: int = 1000
    ):
        """Initialize Anthropic provider"""
        import anthropic

        self.client = anthropic.Anthropic() # Uses ANTHRPOPIC_API_KEY env var
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


    def generate(self, prompt: str) -> str:
        """Generate response from Anthropic"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            message=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant that answers questions about incident reviews. Be concise and cite your sources."
        )

        return message.content[0].text
    

class LocalOllamaProvider:
    """
    Local Ollama integration for offline use

    Ollama lets you run models like Llama and Mistral locally
    Install from: https://ollama.ai
    """

    def __init__(
            self,
            model: str = 'llama3.2',
            base_url: str = 'http: //localhost:11434'
    ):
        """Initialize Ollama providers"""
        import requests

        self.model = model
        self.base_url = base_url
        self.session = requests


    def generate(self, prompt: str) -> str:
        """Generate response from Ollama"""
        response = self.session.post(
            f'{self.base_url}/api/generate',
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        response.raise_for_status()
        return response.json()["response"]
    

def get_provider(provider_name: str = 'openai') -> LLMProvider:
    """
    Factory function for LLM providers

    Args:
    provider_name: One of 'openai', 'anthropic', 'ollama'
    """

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": LocalOllamaProvider
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return providers[provider_name]