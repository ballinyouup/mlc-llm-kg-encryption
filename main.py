from mlc_llm import MLCEngine

# Create engine
model = "./models/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC"
engine = MLCEngine(model, device="metal")

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        model=model,
        stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()