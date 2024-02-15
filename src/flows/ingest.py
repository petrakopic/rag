from functools import lru_cache

import spacy
from tqdm import tqdm

import config
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.database.qdrant import LocalQdrantClient
from src.embedding.embed import embed
from src.file_system.pdf import read, create_summary
from src.file_system.utils import remove_header


@lru_cache(maxsize=20)
def run(file_path: str):
    db = LocalQdrantClient(collection_name="embedrai", vector_dim=384)
    # Read and clean the text
    docs = read(file_path)
    for doc in docs:
        doc.text = remove_header(doc.text)

    # Create small sentences using
    text = [d.text for d in docs]
    text = "\n".join(text)
    nlp = spacy.load(config.ENGLISH_SENTENCE_MODEL)
    sentences = list(nlp(text).sents)

    vectors = []
    payloads = []
    ids = []
    _id = 0

    for sentence in tqdm(
        sentences, total=len(sentences), desc="Embedding sentences..."
    ):
        _id += 1
        vectors.append(embed(sentence.text))
        payloads.append({"text": sentence.text})
        ids.append(_id)

    del sentences
    del text

    for doc in tqdm(docs, total=len(docs), desc="Embedding documents parts..."):
        _id += 1
        vectors.append(embed(doc.text))
        payload = {"text": doc.text}
        if "table" in doc.text:
            payload["table"] = "True"
        if contains_person(doc.text):
            payload["person"] = ["True"]
        payloads.append(payload)
        ids.append(_id)

    del docs

    summaries = create_summary(file_path)
    for summary_part in tqdm(
        summaries, total=len(summaries), desc="Embedding summaries"
    ):
        _id += 1
        vectors.append(embed(summary_part))
        payload = {"text": summary_part}
        payloads.append(payload)
        ids.append(_id)

    db.ingest_vectors(vectors=vectors, payloads=payloads, ids=ids)
    print(f"Sucesfully ingested {len(vectors)} vectors in total.")


def contains_person(text: str) -> bool:
    """
    Check whether a person is mentioned in the text
    """

    tokenizer = AutoTokenizer.from_pretrained(config.ENTITY_RECOGNITION_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        config.ENTITY_RECOGNITION_MODEL
    )

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    for result in ner_results:
        if (
            result["entity"] == "B-PER"
            or result["entity"] == "I-PER"
            and result["score"] > 0.9
        ):
            return True
    return False


def contains_table_description():
    pass
