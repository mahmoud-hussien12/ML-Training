import os
from openai import OpenAI

class Generator:
    def __init__(
        self, 
        model_name: str = "gpt-4.1-mini", 
        temperature: float = 0, 
        max_tokens: int = 300
    ):
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:11434/v1")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    