import json

from mlc_llm.contrib.embeddings.embeddings import MLCEmbeddings


def fix_missing_predicate_object(triple):
    if not isinstance(triple, dict):
        return triple
    keys = list(triple.keys())
    values = list(triple.values())

    # 2-key pattern: {"subject": "user", "praises": "product"}
    # the unknown key IS the predicate, its value IS the object
    if len(keys) == 2 and keys[0] == "subject" and keys[1] not in ("predicate", "object"):
        return {"subject": values[0], "predicate": keys[1], "object": values[1]}

    # 3-key pattern with subject first but wrong key names for predicate/object
    # e.g. {"subject": "user", "praises": "scent", "object": "gift"}
    #   or {"subject": "product", "wants_for": "is", "object": "this_product"}
    if len(keys) == 3 and keys[0] == "subject":
        if keys[1] != "predicate" or keys[2] != "object":
            return {"subject": values[0], "predicate": values[1], "object": values[2]}

    return triple


def extract_unique_strings(triples):
    entities = set()
    predicates = set()

    for triple in triples:
        entities.add(triple["subject"])
        entities.add(triple["object"])
        predicates.add(triple["predicate"])

    return list(entities), list(predicates)


NOISY_PREDICATES = {
    "predicate",
    "predicateicate",
    "predate",
    "prediate",
    "praicize",
    "praize",
    "has_properforms_feature",
}

GENERIC_TERMS = {
    "user",
    "product",
    "review",
    "it",
    "this_product",
}


def is_noisy_predicate(predicate):
    if predicate in NOISY_PREDICATES:
        return True
    # Catch typo variants like "predicateicate", "prediacte", etc.
    if "prediat" in predicate or ("predicate" in predicate and predicate != "predicate"):
        return True
    return False


def tokenize_term(term):
    return [token for token in term.lower().replace("_", " ").split() if token]


def is_grounded_term(term, text_blob):
    if term in GENERIC_TERMS:
        return True
    term_tokens = tokenize_term(term)
    if not term_tokens:
        return False
    # require meaningful token overlap to avoid keeping hallucinated entities
    return any(len(token) >= 3 and token in text_blob for token in term_tokens)


def is_grounded_triple(triple, review_text, review_title):
    text_blob = f"{review_title} {review_text}".lower()
    subject = triple["subject"].strip().lower()
    obj = triple["object"].strip().lower()
    return is_grounded_term(subject, text_blob) or is_grounded_term(obj, text_blob)

def is_valid_triple(triple):
    return (
        isinstance(triple, dict)
        and isinstance(triple.get("subject"), str) and triple["subject"] != ""
        and isinstance(triple.get("predicate"), str) and triple["predicate"] != ""
        and isinstance(triple.get("object"), str) and triple["object"] != ""
        and not is_noisy_predicate(triple["predicate"])
    )


# run the following command within the model folder to compile the model
# mlc_llm compile ./mlc-chat-config.json --model-type bert --device metal -o lib.so
def normalize():
    processed_reviews = []
    with open("./ministral_3b_outputs/amazon_train_output_split_0.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            triples = data.get("triples")

            # skip the review entirely if there are no triples or triples is not a list
            if not isinstance(triples, list) or len(triples) == 0:
                continue

            # attempt to fix missing predicate/object keys, then filter malformed triples
            fixed_triples = []
            seen_triples = set()
            review_text = data.get("text", "")
            review_title = data.get("title", "")
            for triple in triples:
                triple = fix_missing_predicate_object(triple)
                if not is_valid_triple(triple):
                    continue
                if not is_grounded_triple(triple, review_text, review_title):
                    continue

                triple_key = (
                    triple["subject"].strip().lower(),
                    triple["predicate"].strip().lower(),
                    triple["object"].strip().lower(),
                )
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)
                fixed_triples.append(triple)

            # skip the review if no valid triples remain after filtering
            if len(fixed_triples) == 0:
                continue

            # preserve all review metadata, just replace triples with the cleaned ones
            data["triples"] = fixed_triples
            processed_reviews.append(data)

    # write to a file as jsonl, preserving full review data
    with open("./ministral_3b_outputs/amazon_train_output_split_0_processed.jsonl", "w") as f:
        for review in processed_reviews:
            f.write(json.dumps(review) + "\n")

    # we need to look for duplicate nodes like USA, United States of America, US, etc.
    # and normalize them to a single node
    # one strategy is to use the embeddings to find the most similar nodes
    # and then merge them
    embeddings_model = MLCEmbeddings(
        model="./models/snowflake-arctic-embed-s-q0f32-MLC",
        model_lib_path="./models/snowflake-arctic-embed-s-q0f32-MLC/lib.so"
    )

    # load in triples
    triples = []
    with open("./ministral_3b_outputs/amazon_train_output_split_0.jsonl", "r") as f:
        for line in f:
            triples.append(json.loads(line))
    # print(embeddings_model.embed(["Hello"]))

if __name__ == "__main__":
    normalize()