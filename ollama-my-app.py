import ollama

# =========================================================
# CONFIGURATION
# =========================================================

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

DATASET_FILE = 'mobile-data.txt'

# =========================================================
# LOAD DATASET
# =========================================================

with open('mobile-details.txt', 'r', encoding='utf-8') as file:
    dataset = file.read().split("\n\n\n")

print(f"Loaded {len(dataset)} mobile phone entries")

# =========================================================
# VECTOR DATABASE
# =========================================================

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )['embeddings'][0]

    VECTOR_DB.append((chunk, embedding))

# Generate embeddings for all chunks
for i, chunk in enumerate(dataset):
    chunk = chunk.strip()

    if not chunk:
        continue

    add_chunk_to_database(chunk)

    print(f"Embedded {i+1}/{len(dataset)}")

# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)

# =========================================================
# RETRIEVAL FUNCTION
# =========================================================

def retrieve(query, top_n=5):

    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]

    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]

# =========================================================
# MAIN CHAT LOOP
# =========================================================

while True:

    input_query = input("\nAsk about mobile phones (or type 'exit'): ")

    if input_query.lower() == "exit":
        break

    # Retrieve relevant knowledge
    retrieved_knowledge = retrieve(input_query)

    print("\nRetrieved Knowledge:\n")

    for chunk, similarity in retrieved_knowledge:
        print(f"Similarity: {similarity:.2f}")
        print(chunk[:300])
        print("-" * 50)

    # Build context
    context = "\n".join(
        [f"- {chunk}" for chunk, similarity in retrieved_knowledge]
    )

    # System prompt
    instruction_prompt = f"""
You are an expert smartphone recommendation assistant.

Answer ONLY using the provided context.

Rules:
- Do not make up specifications.
- If information is unavailable, say "I don't know".
- Recommend phones based on user needs.
- Mention pros and cons when possible.
- Compare devices clearly.

Context:
{context}
"""

    print("\nAssistant:\n")

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {
                'role': 'system',
                'content': instruction_prompt
            },
            {
                'role': 'user',
                'content': input_query
            },
        ],
        stream=True,
    )

    full_response = ""

    for chunk in stream:
        token = chunk['message']['content']
        full_response += token
        print(token, end='', flush=True)

    print("\n")