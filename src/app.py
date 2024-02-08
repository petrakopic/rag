import streamlit as st
from tqdm import tqdm

from src.database.qdrant import LocalQdrantClient
from src.embedding.embed import embed
from src.file_system.pdf import read
from src.llm.prompt import create_hyde, generate_context_around_rag, rewrite_retrieve_read

st.title("Mini RAG Chatbot")


file_name = st.sidebar.text_input(
    label="#### Your PDF directory path: ",
    placeholder="C:\data",
    help="Enter the directory path to your PDF file.",
    type="default")


def run(query, file_name):
    db = LocalQdrantClient(collection_name="TEST", vector_dim=384)

    docs = read(file_name)
    for idx, chunk in tqdm(enumerate(docs), desc="Embedding in process...", total=len(docs)):
        chunk.embedding = embed(chunk.text)

    vectors = []
    payloads = []
    ids = []

    for doc in docs:
        vectors.append(doc.embedding)
        payloads.append(doc.text)
        ids.append(doc.id_)

    db.ingest_vectors(vectors=vectors, payloads=payloads, ids=ids)

    # 1 try to find the answer without changing the query
    response = db.retrieve_vector(vector = embed(query))[0]
    context, score = response.payload["text"], response.score

    if score > 0.7:
        answer = generate_context_around_rag(question=query, answer=context)
        return answer


    # 2 try to find the answer by changing the query
    query = create_hyde(query=query)
    response = db.retrieve_vector(vector=embed(query))[0]
    context, score = response.payload["text"], response.score

    if score > 0.7:
        answer = generate_context_around_rag(question=query, answer=context)
        return answer

    #
    query = rewrite_retrieve_read(query)
    response = db.retrieve_vector(vector=embed(query))[0]
    context, score = response.payload["text"], response.score

    if score > 0.7:
        answer = generate_context_around_rag(question=query, answer=context)
        return answer

    st.write("Sorry, I couldn't find an answer to your question.")


query = st.text_input("Your question", help="Enter your question here.")

if st.button("Ask"):
    if not query.strip() or not file_name.strip():
        st.error("Please provide both the directory path and the search query.")
    else:
        try:
            # Assuming run is your function to process the query and file_name
            answer = run(query, file_name)
            st.write(answer.content)
        except Exception as e:
            st.error(f"An error occurred: {e}")
