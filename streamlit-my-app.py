import streamlit as st
import ollama
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mobile Phone RAG Assistant",
    layout="wide"
)

st.title("📱 Mobile Phone RAG Assistant")

st.write(
    "Ask questions about smartphones, gaming phones, "
    "camera phones, battery life, and comparisons."
)

# =========================================================
# MODEL CONFIG
# =========================================================

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

DATASET_FILE = 'mobile-data.txt'

# =========================================================
# INITIALIZE VECTOR DATABASE
# =========================================================

@st.cache_resource
def initialize_vector_db():

    

    with open('mobile-details.txt', 'r', encoding='utf-8') as file:
        dataset = file.read().split("\n\n\n")

    vector_db = []

    progress_bar = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(dataset):

        chunk = chunk.strip()

        if not chunk:
            continue

        status.text(f"Embedding chunk {i+1}/{len(dataset)}")

        embedding = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunk
        )['embeddings'][0]

        vector_db.append((chunk, embedding))

        progress_bar.progress((i + 1) / len(dataset))

    progress_bar.empty()
    status.empty()

    return vector_db

# Load vector database
with st.spinner("Loading vector database..."):
    VECTOR_DB = initialize_vector_db()

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
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve(query, top_n=5):

    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]

    similarities = []

    for chunk, embedding in VECTOR_DB:

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        similarities.append((chunk, similarity))

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Database Info")

    st.success(f"Loaded {len(VECTOR_DB)} phone entries")

    st.markdown("---")

    st.subheader("Retrieved Context")

    context_placeholder = st.empty()

# =========================================================
# USER INPUT
# =========================================================

input_query = st.text_input(
    "Ask a smartphone question:",
    placeholder="Best gaming phone under 70000?"
)

# =========================================================
# MAIN CHAT LOGIC
# =========================================================

if input_query:

    retrieved_knowledge = retrieve(input_query)

    # Show retrieved chunks in sidebar
    with context_placeholder.container():

        for chunk, similarity in retrieved_knowledge:

            st.markdown(
                f"### Similarity: {similarity:.2f}"
            )

            st.write(chunk[:700])

            st.markdown("---")

    # Build context
    context = "\n".join(
        [f"- {chunk}" for chunk, similarity in retrieved_knowledge]
    )

    # Prompt
    instruction_prompt = f"""
You are an expert smartphone recommendation assistant.

Answer ONLY using the given context.

Rules:
- Never invent specifications.
- If information is unavailable, say "I don't know".
- Recommend phones according to user needs.
- Mention pros and cons.
- Compare devices clearly.

Context:
{context}
"""

    st.subheader("Assistant Response")

    response_placeholder = st.empty()

    full_response = ""

    try:

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

        for chunk in stream:

            token = chunk['message']['content']

            full_response += token

            response_placeholder.markdown(
                full_response + "▌"
            )

        response_placeholder.markdown(full_response)

    except Exception as e:

        st.error(f"Error: {e}")