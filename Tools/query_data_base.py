# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
import chromadb


class queryDataBase:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.client = chromadb.PersistentClient(path="Data Processing Pipeline/chroma_db")
        self.collection = self.client.get_collection(name="website_articles")

    def search(self, query: str):
        """Searches the ChromaDB vector database for articles matching the given query."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        return results
        

        
