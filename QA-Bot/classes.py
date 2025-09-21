from langchain_core.embeddings import Embeddings
import lmstudio as lms

class NewEmbeddings(Embeddings):
    
    def __init__(self):
        self.model = lms.embedding_model("text-embedding-granite-embedding-125m-english")
        
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string"""
        try:
            return self.model.embed(text)
        except Exception as e:
            raise RuntimeError(f"Error in embed_query(): {e}")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents"""
        try:
            return [self.model.embed(t) for t in texts]
        except Exception as e:
            raise RuntimeError(f"Error in embed_documents(): {e}")
