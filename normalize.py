import json

from mlc_llm.contrib.embeddings.embeddings import MLCEmbeddings

# TODO
def normalize():
    embeddings_model = MLCEmbeddings(
        model="./models/snowflake-arctic-embed-s-q0f32-MLC",
        model_lib_path="./models/snowflake-arctic-embed-s-q0f32-MLC/lib.so"
    )

    # load in triples
    triples = []
    with open("./ministral_3b_outputs/amazon_train_output_split_0.jsonl", "r") as f:
        for line in f:
            triples.append(json.loads(line))
    print(embeddings_model.embed(["Hello"]))

if __name__ == "__main__":
    normalize()