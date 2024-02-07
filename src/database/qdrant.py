from tqdm import tqdm
from typing import List
from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct


class LocalQdrantClient:
    def __init__(
        self,
        collection_name: str,
        vector_dim: str,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.client = BaseQdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim, distance=models.Distance.COSINE
            ),
        )

    def ingest_vectors(self, vectors: List[float], payloads: list[str], ids: list[str]):

        if not len(vectors) == len(payloads) == len(ids):
            raise ValueError("Length of vectors, payloads, and ids should be the same.")

        points = []
        for idx, vector in tqdm(
            enumerate(vectors), desc="Loading...", total=len(vectors)
        ):
            points.append(
                PointStruct(id=ids[idx], payload={"text": payloads[idx]}, vector=vector)
            )

        self.client.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )

    def ingest_vector(self, vector: float, payload: str, id: str):
        self.ingest_vectors([vector], [payload], [id])

    def delete_vector(self, point_id: str):
        """
        Delete a vector from the database by its point ID.
        :param point_id: The unique ID of the point to be deleted.
        """
        self.client.delete(
            collection_name=self.collection_name, wait=True, points_selector=[point_id]
        )

    def retrieve_vector(self, vector: list[float]):
        """
        Retrieve a 5 most similar vectors to the given vector.
        :param vector:
        """
        return self.client.search(
            collection_name=self.collection_name,
            search_params=models.SearchParams(hnsw_ef=64, exact=False),
            query_vector=vector,
            limit=5,
        )

    def reranking(self, query_vector: list[float], initial_results: list[str]):
        """
        After first-stage retrieval (lexical/keyword-based search or semantic/embedding-based
        search), doing re-ranking as a second stage to rank retrieved documents using relevance
        scores
        """
        pass

    def calculate_maximal_marginal_relevance(self, query_vector: list[float], initial_results: list[str]):
        """
        There can be similar documents capturing the same information.
        The MMR metric penalizes redundant information """
        pass

    def context_optimization(self, query_vector: list[float], initial_results: list[str]):
        """
        Stuffing/MapReduce/MapRerank
        """
        pass