import streamlit as st
from src.flows import ingest, retrieve

st.title("Mini RAG Chatbot")

# Initialize session state for file_name if not already done
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# Sidebar input for PDF directory path, but only show it if it's not already set
if not st.session_state.file_name:
    file_name = st.sidebar.text_input(
        label="#### Your PDF directory path: ",
        placeholder="C:/data",
        help="Enter the directory path to your PDF file.",
        type="default",
    )

    # Update the session state with the file name when it's entered
    if file_name:
        st.session_state.file_name = file_name


def run(query):
    # Use the file_name from session state
    file_name = st.session_state.file_name

    ingest.run(file_path=file_name)
    answer = retrieve.run(question=query)

    if answer:
        st.write("Answer found:")
        st.write(answer.content)
    else:
        st.write("Sorry, I couldn't find an answer to your question.")


query = st.text_input("Your question", help="Enter your question here.")

if st.button("Ask"):
    if not query.strip() or not st.session_state.file_name.strip():
        st.error("Please provide both the directory path and the search query.")
    else:
        try:
            # Assuming run is your function to process the query
            run(query)
        except Exception as e:
            st.error(f"An error occurred: {e}")
