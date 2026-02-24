from mlc_llm.contrib.embeddings.embeddings import MLCEmbeddings
# run the following command within the model folder to compile the model
# mlc_llm compile ./mlc-chat-config.json --model-type bert --device metal -o lib.so
def normalize():
    emb = MLCEmbeddings(
        model="./models/snowflake-arctic-embed-s-q0f32-MLC",
        model_lib_path="./models/snowflake-arctic-embed-s-q0f32-MLC/lib.so"
    )
