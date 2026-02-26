from mlc_llm import AsyncMLCEngine
from openai import OpenAI
import os
from pydantic import BaseModel
from dotenv import load_dotenv

class Engine(AsyncMLCEngine):
    def __init__(self, model_path, device):
        super().__init__(model=model_path, device=device)
        self.engine_config.mode = "server"

    def send_json_message(self, message, temperature=0.1, max_tokens=4096, frequency_penalty=0.1, presence_penalty=0.1):
        return self.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            stream=False,
            response_format={"type": "json_object",
                             "json_schema": extract_schema},
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    def send_chat_message(self, message, temperature=0.7, max_tokens=None):
        return self.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def send_extract_message(self, message, temperature=0.5, max_tokens=4096):
        self.set_system_prompt(extract_system_prompt)
        return self.send_json_message(message, temperature, max_tokens)

    def set_system_prompt(self, prompt):
        self.conv_template.system_message = prompt


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class TripleExtraction(BaseModel):
    triples: list[Triple]


class CloudEngine:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set")

        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = None
        self.site_name = None
        self.model = "google/gemini-3-flash-preview"

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _get_extra_headers(self):
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-OpenRouter-Title"] = self.site_name
        return headers if headers else None

    def send_extract_message(self, message, temperature=0.5, max_tokens=4096, frequency_penalty=0.1, presence_penalty=0.1):
        extra_headers = self._get_extra_headers()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": extract_system_prompt}, {"role": "user", "content": message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "response_format": TripleExtraction,
        }

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        return self.client.beta.chat.completions.parse(**kwargs)

    def send_chat_message(self, message, temperature=0.7, max_tokens=None):
        extra_headers = self._get_extra_headers()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        return self.client.chat.completions.create(**kwargs)

extract_system_prompt = """You are an expert multilingual knowledge graph extraction system. Your task is to extract entities and relationships from reviews and output them as a strict JSON array of triples.

Rules:
1. Core Entities: Always use the exact word "user" to represent the person writing the review, and the exact word "product" to represent the item, hotel, or service being reviewed.
2. Entity Normalization: Keep other entities short (1 to 3 words). Format them in lowercase and use underscores for spaces (e.g., "coconut_oil", "front_desk").
3. Multilingual Unification: If the input is in Portuguese or another language, you MUST translate all extracted entities and predicates into English.
4. Predicate Standardization: Use clear, consistent predicates. (e.g., "loves", "criticizes", "has_feature", "is_used_for", "experienced_issue", "wants_in"). NOTE: A "product" cannot criticize or love; only a "user" can perform those actions.
5. Output format: Output ONLY valid JSON. Object must be a string.

Output format:
{
  "triples": [
    {"subject": "entity_1", "predicate": "relationship", "object": "entity_2"}
  ]
}

Examples:

Text: "waste of my money. very thin & too small for my fingernails & toenails"
Output:
{
  "triples": [
    {"subject": "product", "predicate": "is", "object": "waste_of_money"},
    {"subject": "product", "predicate": "has_attribute", "object": "very_thin"},
    {"subject": "user", "predicate": "criticizes", "object": "too_small"},
    {"subject": "product", "predicate": "is_used_for", "object": "fingernails"},
    {"subject": "product", "predicate": "is_used_for", "object": "toenails"}
  ]
}

Text: "We stayed here for a friends wedding and fell in Love. The hotel is right on one of the main pedestrian roads and within minutes of all the glory that Vail has to offer. The room was cute, super fluffy and comfortable bed. Tiny windows which made the room a bit dark, but not a big deal."
Output:
{
  "triples": [
    {"subject": "user", "predicate": "stayed_for", "object": "wedding"},
    {"subject": "product", "predicate": "located_on", "object": "pedestrian_road"},
    {"subject": "product", "predicate": "located_near", "object": "vail"},
    {"subject": "product", "predicate": "has_feature", "object": "cute_room"},
    {"subject": "product", "predicate": "has_feature", "object": "comfortable_bed"},
    {"subject": "product", "predicate": "has_attribute", "object": "tiny_windows"}
  ]
}

Text: "Chegou antes do prazo, ótima qualidade, fácil montagem e super pratico de usar!"
Output:
{
  "triples": [
    {"subject": "user", "predicate": "loves", "object": "fast_delivery"},
    {"subject": "product", "predicate": "has_attribute", "object": "high_quality"},
    {"subject": "product", "predicate": "has_attribute", "object": "easy_assembly"},
    {"subject": "product", "predicate": "has_attribute", "object": "practical_to_use"}
  ]
}
"""

extract_schema = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                    },
                    "predicate": {
                        "type": "string",
                    },
                    "object": {
                        "type": "string",
                    }
                },
                "required": ["subject", "predicate", "object"],
            }
        }
    },
    "required": ["triples"],
}
