from pathlib import Path
from llama_index import Document
from llama_hub.file.pdf.base import PDFReader


def read(file_path: str) -> list[Document]:
    """Read the content of a PDF file and return it as a list of Documents."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")

    loader = PDFReader()
    docs = loader.load_data(file=path)
    return docs
