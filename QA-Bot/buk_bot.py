
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# class QABot:
#     def __init__(self, retriever, model_name: str = "gpt-4o-mini"):
#         self.retriever = retriever
#         self.llm = ChatOpenAI(model=model_name)

#     def ask(self, query: str):
#         from langchain.chains import RetrievalQA

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             retriever=self.retriever,
#             chain_type="stuff"
#         )
#         return qa_chain.run(query)

# class VectorStore:
#     def __init__(self, embedder: Embedder):
#         self.embedder = embedder
#         self.db = None

#     def build(self, texts: list[str]):
#         self.db = FAISS.from_texts(texts, self.embedder.embedding_model)

#     def get_retriever(self, k: int = 3):
#         return self.db.as_retriever(search_kwargs={"k": k})

# class TextProcessor:
#     def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap
#         )

#     def split(self, text: str):
#         return self.splitter.split_text(text)


# class Embedder:
#     def __init__(self, model_name: str):
#         self.embedding_model = OpenAIEmbeddings(model=model_name)

#     def embed(self, texts: list[str]):
#         return self.embedding_model.embed_documents(texts)


# class DocumentLoader:
#     def __init__(self, file_path: str):
#         self.file_path = file_path
    
#     def load(self) -> str:
#         with open(self.file_path, "r", encoding="utf-8") as f:
#             return f.read()




# if __name__ == "__main__":
#     # Load text
#     loader = DocumentLoader("sample.txt")
#     text = loader.load()

#     # Process text
#     processor = TextProcessor()
#     chunks = processor.split(text)

#     # Embeddings + VectorStore
#     embedder = Embedder()
#     store = VectorStore(embedder)
#     store.build(chunks)

#     # Q&A Bot
#     retriever = store.get_retriever()
#     bot = QABot(retriever)

#     # Ask questions
#     while True:
#         query = input("Ask a question (or 'exit'): ")
#         if query.lower() == "exit":
#             break
#         answer = bot.ask(query)
#         print("Answer:", answer)


from bot import BOT

bot = BOT('filename.txt')
value = bot.query("Who is the background of Robert Harmon")
print(value)