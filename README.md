## RAG 

This repository contains the implementation of a custom Retrieval-Augmented Generation (RAG) system designed for 
efficient handling and querying of files. It allows users to upload PDF files and ask questions about their content 
through an easy-to-use Streamlit application.


## High level overview of the architecture
We leverage pdfreader to intelligently segment the PDF into manageable chunks by understanding the document's layout. 
The entire text is then consolidated and segmented into sentences using the spaCy English sentence model. For every 
segment, we generate summaries (Falconsai model) to improve the  retrieval of high-level concepts.

Text Cleaning: Our process involves cleaning the document to strip away headers and junk characters picked up 
during the PDF reading stage (for RAG it holds - garbage in, garbage out ;D).

Since our use case demend to extract information about individuals, we use the bert-base-NER model to identify
the entities in the text. We then store the entities in a Qdrant database with a special flag in the metadata field.


Vector Embedding and Storage: We use the BAAI/bge-small-en-v1.5 model for text embedding into vectors and utilize 
Qdrant DB for storage. Ewerything runs on a 8-core CPU with 16GB RAM setup.

Intelligent Query Management: we implemented custom query to assess the nature of user inquiries (person-related or general) 
and tailor the vector retrieval accordingly. 
If no context was found, we use the HYDE method rewrite_retrieve_read method for enhanced retrieval accuracy. 

Contextual Answer generation: we utilize GPT-3.5 

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (for Qdrant database)
- Access to a ChatGPT API key (for the LLM model), saved in the environment variable `OPENAI_API_KEY`

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
