from langchain_core.embeddings import Embeddings
import lmstudio as lms


class NewEmbeddings(Embeddings):
    
    def __init__(self):
        self.embeddings = lms.embedding_model("text-embedding-granite-embedding-125m-english")
        
    def embed(self, text):
        try:
            embed = self.embeddings.embed(text)
            return embed
        except Exception :
            raise NotImplementedError("Something went wrong in embed()")
    
    def embed_query(self, text):
        try:
            embed = self.embeddings.embed(text)
            return embed
        except Exception :
            raise NotImplementedError("Something went wrong in embed_query()")
    
    def embed_documents(self, text):
        try:
            embed = self.embeddings.embed(text)
            return embed
        except Exception :
            raise NotImplementedError("Something went wrong in embed_documents()")
    
