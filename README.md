## RAG 

This repository contains the implementation of a custom Retrieval-Augmented Generation (RAG) system designed for 
efficient handling and querying of files. It allows users to upload PDF files and ask questions about their content 
through an easy-to-use Streamlit application.
The system process the files locally, ingests them into the local Qdrant database, and uses the LLM model for 
interaction.


## Getting Started

### Prerequisites
- Python 3.8+
- Docker (for Qdrant database)
- Access to a ChatGPT API key (for the LLM model)


### Installation

1. Clone the repository
```bash
git clone git@github.com:petrakopic/rag.git
```

2. Install the requirements
```bash
pip install -r requirements.txt
```

3. Run Qdrant database using docker-compose
```bash
docker run -p 6333:6333 qdrant/qdrant
```

4. Run the app 
Run the streamlit app
```bash
streamlit run app.py
```