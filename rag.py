import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Load ALL txt files
txt_loader = DirectoryLoader(
    "data/",
    glob="*.txt",
    loader_cls=TextLoader
)
pdf_loader = DirectoryLoader(

    "data/",

    glob="*.pdf",
    loader_cls=PyMuPDFLoader

)

# Combine documents
documents = txt_loader.load() + pdf_loader.load()

print(f"Loaded {len(documents)} documents")
# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs = splitter.split_documents(documents)
docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]
print(f"Created {len(docs)} chunks")

# 3. Embeddings (local)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 4. Vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 6. Ollama LLM
llm = Ollama(model="gemma3:4b")  # or qwen2.5:7b

# 7. Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below and if the answer is not contained within the context, say 'I don't know'. :
                                          

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

    # -----------------------------
    # DEBUG: SHOW RETRIEVED CHUNKS
    # -----------------------------
    retrieved_docs = retriever.invoke(query)

    print("\n" + "="*80)
    print("RETRIEVED CHUNKS")
    print("="*80)

    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i+1}")
        print("-"*80)

        # show source if available
        if "source" in doc.metadata:
            print(f"Source: {doc.metadata['source']}")

        # show page number if PDF
        if "page" in doc.metadata:
            print(f"Page: {doc.metadata['page']}")

        print("\nContent:\n")
        print(doc.page_content)

        print("\n" + "="*80)

    # -----------------------------
    # GENERATE ANSWER
    # -----------------------------
    answer = rag_chain.invoke(query)

    print("\nAnswer:\n", answer)