from langchain_community.vectorstores import FAISS

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import faiss
import re

import lmstudio as lms

# model = lms.embedding_model("text-embedding-granite-embedding-125m-english")


# embedding = model.embed("Hello, world!")

# print(embedding)

# text_input = "hello this is Harsh"

# data = embeddings.embed_query(text = text_input)
# print(data)


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    base_url = "http://127.0.0.1:1234/v1",
    model = "google_gemma-3-270m-it",
    api_key = "no neend"
)

print(llm.invoke("hi"))