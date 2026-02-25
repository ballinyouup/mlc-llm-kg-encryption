from mlc_llm import AsyncMLCEngine
from enum import IntEnum


# enum for model choice
class ModelChoice(IntEnum):
    ministral_3b = 0
    phi_4_mini = 1
    mistral_7b = 2
    llama_3_2_3b = 3


class Engine(AsyncMLCEngine):
    model_choice = ModelChoice(0)
    models = ["./models/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC", "./models/Phi-4-mini-instruct-q4f32_1-MLC", "./models/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    "./models/Llama-3.2-3B-Instruct-q4f32_1-MLC"]

    def __init__(self, device, model_choice=model_choice):
        super().__init__(model=Engine.models[model_choice], device=device)
        self.engine_config.mode = "server"

    def send_json_message(self, message, temperature=0.1, max_tokens=2048, frequency_penalty=0.1, presence_penalty=0.1):
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

    def send_extract_message(self, message, temperature=0.3, max_tokens=2048):
        self.set_system_prompt(extract_system_prompt)
        return self.send_json_message(message, temperature, max_tokens)

    def set_system_prompt(self, prompt):
        self.conv_template.system_message = prompt


extract_system_prompt = """You are an expert knowledge graph extraction system. Your task is to extract entities and relationships from Amazon product reviews and output them as a strict JSON array of triples.

Rules:
1. Extract specific entities (products, features, materials, body parts, users, etc.).
2. Keep entities short (1 to 3 words) and normalize them (e.g., lowercase, use underscores for spaces like "coconut_oil").
3. Use ONLY clear, specific predicates.
4. Do not extract stop words or conversational filler.
5. Output ONLY valid JSON.
6. Object must be a string. Do not include additional properties.

Output format:
{
  "triples": [
    {"subject": "entity_1", "predicate": "relationship", "object": "entity_2"}
  ]
}

Examples:

Text: "Love the faux nails and so happy to find glitter nails for toes!"
Output:
{
  "triples": [
    {"subject": "user", "predicate": "praises", "object": "faux_nails"},
    {"subject": "faux_nails", "predicate": "has_feature", "object": "glitter"},
    {"subject": "faux_nails", "predicate": "is_used_for", "object": "toes"}
  ]
}

Text: "waste of my money. very thin & too small for my fingernails & toenails"
Output:
{
  "triples": [
    {"subject": "product", "predicate": "is", "object": "waste_of_money"},
    {"subject": "product", "predicate": "has_attribute", "object": "very_thin"},
    {"subject": "product", "predicate": "criticizes", "object": "too_small"},
    {"subject": "product", "predicate": "is_used_for", "object": "fingernails"}
  ]
}

Text: "Loved the scent but want this product in a perfume too!"
Output:
{
  "triples": [
    {"subject": "user", "predicate": "praises", "object": "scent"},
    {"subject": "product", "predicate": "has_attribute", "object": "scent"},
    {"subject": "user", "predicate": "wants", "object": "perfume"}
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
