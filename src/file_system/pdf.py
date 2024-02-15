from pathlib import Path
from llama_index import Document
from llama_hub.file.pdf.base import PDFReader
from PyPDF2 import PdfReader
from transformers import pipeline
from tqdm import tqdm
from config import TEXT_SUMMARIZATION_MODEL


def read(file_path: str) -> list[Document]:
    """Read the content of a PDF file and return it as a list of Documents."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")

    loader = PDFReader()
    docs = loader.load_data(file=path)

    return docs


def read_by_pages(file_path: str) -> list[str]:
    """Read the content of a PDF file and return it as a list of Documents."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)
    return pages


def create_summary(file_path: str) -> list[str]:
    pages = read_by_pages(file_path)
    summarizer = pipeline("summarization", model=TEXT_SUMMARIZATION_MODEL)

    summary_parts = []

    for page in tqdm(pages, total=len(pages), desc="Summarizing..."):
        summary = summarizer(page, max_length=100, min_length=30, do_sample=False)
        summary_parts.append(summary[0]["summary_text"])

    return summary_parts
