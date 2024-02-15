import config
from src.database.qdrant import LocalQdrantClient
from src.embedding.embed import embed
from src.llm.prompt_engineering import (
    router_query,
    rewrite_retrieve_read,
    generate_context_around_rag,
)


def run(question: str):
    db = LocalQdrantClient(collection_name=config.QDRANT_COLLECTION, vector_dim=384)

    # Route the question to the right subset of the database
    router = router_query(question)
    if router == "person":
        context = db.retrieve_with_filtering(
            embed(question), key="person", value="True"
        )
        if not context:
            question = rewrite_retrieve_read(question)
            return
            # TODO: test recursive call
            # return run(question)
        return generate_context_around_rag(
            question=question, answer=[a.payload["text"] for a in context]
        )

    elif router == "table":
        context = db.retrieve_with_filtering(embed(question), key="table", value="True")
        if not context:
            # TODO: test recursive call
            question = rewrite_retrieve_read(question)
            return
        return generate_context_around_rag(
            question=question, answer=[a.payload["text"] for a in context]
        )

    else:
        context = db.retrieve_vector(vector=embed(question))
        if context[0].score > 0.6:
            return generate_context_around_rag(
                question=question, answer=[a.payload["text"] for a in context]
            )
        else:
            # TODO: test recursive call
            question = rewrite_retrieve_read(question)
            # return run(question)
            return
