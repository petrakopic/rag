import streamlit as st
import os
import tempfile
from llama_index import SimpleDirectoryReader

st.title("Mini RAG Chatbot")


file_name = st.sidebar.text_input(
    label="#### Your PDF directory path: ",
    placeholder="C:\data",
    help="Enter the directory path to your PDF file.",
    type="default")


def run(query, file_name):

    # Your code here
    pass


query = st.text_input("Your question", help="Enter your question here.")

if st.button("Ask"):
    if not query.strip() or not file_name.strip():
        st.error("Please provide both the directory path and the search query.")
    else:
        try:
            # Assuming run is your function to process the query and file_name
            run(query, file_name)
        except Exception as e:
            st.error(f"An error occurred: {e}")
