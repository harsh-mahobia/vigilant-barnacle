from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import faiss
import lmstudio as lms
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document






#CLASS TO BYPASS THE EMBEDDING.embed_query() FUNCTION
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        pass;
        raise NotImplementedError("We already have embeddings, no need to compute.")
    
    def embed_query(self, text):
        pass;
        raise NotImplementedError("Use similarity_search_by_vector instead.")



class BOT:

    def __init__(self, url : str):
        self.url = url
        self.llm = ChatOpenAI(
            base_url = "http://127.0.0.1:1234/v1",
            model = "qwen2-500m-instruct",
            api_key = "no neend"
        )
            # FETCHING TRANSCRIPT IN English
        data = YouTubeTranscriptApi().fetch(url, languages=['en'])
        transcript = " ".join(chunk.text for chunk in data)

            # SPLITTING THE TEXT INTO CHUNKS
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([transcript])
        texts = [chunk.page_content for chunk in chunks]

            # EMBEDDING GENERATOR MODEL
        self.embeddings = lms.embedding_model("text-embedding-granite-embedding-125m-english")

            # Vector Store 
        vectors = self.embeddings.embed(texts)
        self.vector_store = FAISS.from_embeddings(text_embeddings=list(zip(texts, vectors)), embedding=DummyEmbeddings())

    

    def query(self, text : str):
        query = text
        query_vector = self.embeddings.embed([query])[0]   # embed manually
        searched_docs = self.vector_store.similarity_search_by_vector(query_vector, k=4)
        context_text = "\n".join(doc.page_content for doc in searched_docs)
        
        prompt = PromptTemplate.from_template("You are knowledgeable Assitant, Answer the question : {question} \n\n from the below given context : \n{context}")

        parser = StrOutputParser()

        chain = prompt | self.llm | parser

        answer = chain.invoke({"question" : query, "context" : context_text[:200]})

        return answer
        

