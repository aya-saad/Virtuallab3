"""
LLM integration module for connecting to various LLM providers
"""
import os
import logging
import json
import requests
from django.conf import settings


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt, max_tokens=1000, temperature=0.0):
        """Generate text from the LLM"""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIProvider(LLMProvider):
    """OpenAI chat completion provider"""

    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt, max_tokens=1000, temperature=0.0):
        """Generate text from OpenAI"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            messages = [
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on context provided."},
                {"role": "user", "content": prompt}
            ]

            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                self.api_base,
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )

            if response.status_code != 200:
                logging.error(f"Error from OpenAI API: {response.text}")
                return f"Error: Unable to generate response. Status code: {response.status_code}"

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logging.error(f"Error generating text with OpenAI: {str(e)}")
            return f"Error: {str(e)}"


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider"""

    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, endpoint=None):
        super().__init__(model_name, api_key)
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        self.api_base = f"{self.endpoint}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"

    def generate(self, prompt, max_tokens=1000, temperature=0.0):
        """Generate text from Azure OpenAI"""
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }

            messages = [
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on context provided."},
                {"role": "user", "content": prompt}
            ]

            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                self.api_base,
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )

            if response.status_code != 200:
                logging.error(f"Error from Azure OpenAI API: {response.text}")
                return f"Error: Unable to generate response. Status code: {response.status_code}"

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logging.error(f"Error generating text with Azure OpenAI: {str(e)}")
            return f"Error: {str(e)}"


def get_llm_provider(provider="openai", model=None):
    """Factory function to get LLM provider"""

    if provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        model_name = model or os.environ.get("AZURE_OPENAI_MODEL", "gpt-3.5-turbo")
        return AzureOpenAIProvider(model_name=model_name, api_key=api_key, endpoint=endpoint)
    else:  # default to OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model_name = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        return OpenAIProvider(model_name=model_name, api_key=api_key)