import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Load ALL txt files
loader = DirectoryLoader(
    "data/",
    glob="*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 2. Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f"Created {len(docs)} chunks")

# 3. Embeddings (local)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 4. Vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Ollama LLM
llm = Ollama(model="gemma3:4b")  # or qwen2.5:7b

# 7. Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below and if the answer is not contained within the context, say 'I don't know'. Keep the answer concise and to the point. :
                                          

{context}

Question: {question}
""")

# helper to format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. RAG pipeline
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# 9. Chat loop
while True:
    query = input("\nAsk your question (or type 'exit'): ")
    
    if query.lower() == "exit":
        break
    
    answer = rag_chain.invoke(query)
    print("\nAnswer:\n", answer)