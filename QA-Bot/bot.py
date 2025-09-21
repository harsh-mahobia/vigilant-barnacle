from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from classes import NewEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import faiss
from langchain_community.vectorstores import FAISS


class BOT:

    prompt = PromptTemplate(
        template = """You are a knowledgeable assistant. 
            Use the following context to answer the user's question in breif and detailed manner. 
            If the answer cannot be found in the context, say "I don't understand your questions." 

            Context:
            {context}

            Question: {question}

            Answer:""",
        input_variables = ["context", "question"]
    )
    def __init__(self, text):
        self._initialize_chat_model()
        self._initialize_model()
        self.embeddings = NewEmbeddings()
        self._split(text)
        self._create_store()

    def query(self, question):
        context = self._get_context(question)
        return self.model.invoke({'question' : question, 'context' : context})
          
    def _initialize_chat_model(self):
        self.llm = ChatOpenAI(
            base_url = "http://127.0.0.1:1234/v1",
            model = "qwen2-500m-instruct",
            api_key = "no neend"
        )
    
    def _initialize_model(self):
        self.model = self.prompt | self.llm | StrOutputParser()


    def _split(self, text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_overlap=50, chunk_size=200)
        self.docs = splitter.create_documents([text])
        
    
    def _create_store(self):
        self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
    
    def _get_context(self, question):
        query_vector = self.embeddings.embed_query(question)
        searched_docs = self.vector_store.similarity_search_by_vector(query_vector, k=3)
        context_text = "\n".join(doc.page_content for doc in searched_docs)
        return context_text

    

        
    

#########################################################33
## simple skeleton flow


# #load docs
# loader = TextLoader('filename.txt', encoding='utf-8')

# #split the loaded(loader) to docs
# splitter = RecursiveCharacterTextSplitter(chunk_overlap=50, chunk_size=200)
# docs = loader.load_and_split(splitter)


# #embedding Generator
# embeddings = NewEmbeddings()
# vector_store = FAISS.from_documents(docs, embeddings)





# query = "Who is the background of Robert Harmon"

# query_vector = embeddings.embed([query])[0]   # embed manually
# searched_docs = vector_store.similarity_search_by_vector(query_vector, k=5)
# context_text = "\n".join(doc.page_content for doc in searched_docs)

# print(context_text)
# prompt = PromptTemplate(
#     template = """You are a knowledgeable assistant. 
#         Use the following context to answer the user's question. 
#         If the answer cannot be found in the context, say "I don't know." 

#         Context:
#         {context}

#         Question: {question}

#         Answer:""",
#     input_variables = ["context", "question"]
# )
# llm = ChatOpenAI(
#     base_url = "http://127.0.0.1:1234/v1",
#     model = "google_gemma-3-270m-it",
#     api_key = "no neend"
# )

# model = prompt | llm | StrOutputParser()

# answer = model.invoke({'question' : "Who is the Robert Harmon from this document", 'context': context_text})

# print(answer)





# # retriever = vectorstore.as_retriever()
# # # Retrieve the most similar text
# # retrieved_documents = retriever.invoke("Dr. Robert Harmon")

# # # show the retrieved document's content
# # print(retrieved_documents[0].page_content)