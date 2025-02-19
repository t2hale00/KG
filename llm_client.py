from typing import Dict, Any
import os
import json
import requests
from abc import ABC, abstractmethod
import google.generativeai as genai

class LLMClient(ABC):
    @abstractmethod
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a prompt and return structured data."""
        pass

class GoogleAIClient(LLMClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get the model (using the most capable model)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Configure the generation config
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40
        }

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        try:
            # Add explicit instruction for JSON output
            structured_prompt = f"""
            {prompt}
            
            IMPORTANT: Your response must be valid JSON only, with no additional text or explanation.
            If you cannot generate valid JSON, return an empty JSON object {{}}.
            """
            
            # Generate response
            response = self.model.generate_content(
                structured_prompt,
                generation_config=self.generation_config
            )
            
            # Extract the text from the response
            response_text = response.text
            
            # Clean the response text to ensure it's valid JSON
            # Remove any markdown code block indicators
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse and return the JSON
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error processing prompt with Google AI: {str(e)}")
            return {}

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.api_url = "https://api.anthropic.com/v1/messages"

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "claude-3-opus-20240229",
            "max_tokens": 4000,
            "temperature": 0.1,
            "response_format": {"type": "json"}
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            content = response.json()["content"][0]["text"]
            return json.loads(content)
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            return {}

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4-turbo-preview",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json"}
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            return {}

# Example usage:
"""
# Using Google AI's Gemini
client = GoogleAIClient()
extractor = ThreeGPPLLMExtractor(client)

# Or using Anthropic's Claude
client = AnthropicClient()
extractor = ThreeGPPLLMExtractor(client)

# Or using OpenAI's GPT-4
client = OpenAIClient()
extractor = ThreeGPPLLMExtractor(client)

# Process your document
results = extractor.process_document(your_text)
""" 