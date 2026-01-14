import os
import time
from openai import OpenAI
from pathlib import Path
from api import logger
class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=10.0)
        self.prompt_path = Path("api/llm_service/prompts/churn_explanation.txt")

    def _load_prompt(self) -> str:
        return self.prompt_path.read_text()

    def generate_explanation(self, features: dict, prediction: str) -> str:
        logger.log_info(
            "LLM explanation requested",
            data={
                "features": features,
                "prediction": prediction
            }
        )
        prompt = self._load_prompt() \
            .replace("{{features}}", str(features)) \
            .replace("{{prediction}}", prediction)

        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=120
            )
            elapsed_time = time.time() - start_time
            logger.log_info(
                "LLM explanation generated",
                data={
                    "explanation": response.choices[0].message.content.strip(),
                    "latency_sec": round(elapsed_time, 2)
                }
            )

            usage = response.usage
            logger.log_info(
                "LLM token usage",
                data={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.log_error(
                "LLM explanation failed",
                data={
                    "error": str(e)
                }
            )
            return "Explanation temporarily unavailable."
    
    def should_explain(probability: float) -> bool:
        return probability > 0.6

