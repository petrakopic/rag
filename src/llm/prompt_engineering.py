import os
from functools import lru_cache

from openai.types.chat import ChatCompletionMessage
import config
from openai import OpenAI


def generate_context_around_rag(
    question: str, answer: list[str] | str
) -> ChatCompletionMessage | None:
    """
    Generate a structured context around the question and answer.
    """
    client = _initialize_openai_client()

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Write a detailed context integrating the question and answers "
                    "(split by '\n---------\n') into a coherent explanation.",
                },
                {"role": "user", "content": f"Question: {question}\nAnswers: {answer}"},
            ],
            max_tokens=350,
            temperature=0.7,  # TODO: test different temperatures
        )
        context = response.choices[0].message
        return context
    except Exception as e:
        print(f"An error occurred: {e}")
        return


def rewrite_retrieve_read(query: str):
    """Rewrite-Retrieve-Read - enhance the query before using it to search the database.
    See: https://arxiv.org/pdf/2305.14283.pdf
    """
    client = _initialize_openai_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Provide a better search query for web search engine to answer the given "
                "question, end the queries with ’**’",
            },
            {"role": "user", "content": f"{query}"},
        ],
        max_tokens=350,
    )
    return response.choices[0].message


def create_hyde(query: str):
    """Create 'Hyde' - Hypothetical Document Embeddings. See the paper Precise Zero-Shot Dense
    Retrieval without Relevance Labels (https://arxiv.org/abs/2212.10496)"""
    client = _initialize_openai_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "write a short passage to answer the question.",
            },
            {"role": "user", "content": f"Question {query}"},
        ],
        max_tokens=250,
    )
    return response.choices[0].message


def router_query(question: str):
    client = _initialize_openai_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Assess the nature of each user question. Respond with 'general' for queries"
                "seeking broad information or summaries. Use 'person' for inquiries that "
                "necessitate identifying or mentioning individuals. "
                "answer 'table' for questions that explicitly request information about table.",
            },
            {"role": "user", "content": f"Question: {question}"},
        ],
        max_tokens=350,
    )
    return response.choices[0].message.content


@lru_cache(1)
def _initialize_openai_client() -> OpenAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return client


# TODO: investigate the impact (implement) the following:
#  1)Rewrite-Retrieve-Read
#  2) Hypothetical Document Embeddings (HyDE)
#  3) Follow-up question to condensed/standalone one
#  4) RAG Fusion
#  5) Step-Back Prompting
#  6) Multi Query Retrieval / Expansion
