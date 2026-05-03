# ============================================
# COMPLETE RAG PIPELINE - Built on iPad!
# ============================================

# CELL 1 - Install dependencies
!pip install langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu pypdf transformers accelerate

# CELL 2 - Create sample data
with open("sample.txt", "w") as f:
    f.write("""
    Artificial Intelligence is transforming data engineering.
    RAG stands for Retrieval Augmented Generation.
    Vector databases store embeddings for semantic search.
    Python is the most popular language for AI development.
    Data engineers are well positioned to transition into AI roles.
    """)
print("File created!")

# CELL 3 - Load and chunk
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("sample.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}: {chunk.page_content}")

# CELL 4 - Create embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully!")
print(f"Total vectors stored: {vectorstore.index.ntotal}")

# CELL 5 - Test retrieval
query = "What is RAG?"
results = vectorstore.similarity_search(query, k=2)
print(f"Query: {query}")
print(f"\nTop matching chunks:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}: {result.page_content}")

# CELL 6 - Full RAG with LLM
from transformers import pipeline

generator = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

def rag_answer(question):
    # Step 1 - Retrieve relevant chunks
    results = vectorstore.similarity_search(question, k=2)
    context = " ".join([r.page_content for r in results])
    
    # Step 2 - Generate answer using context
    answer = generator(question=question, context=context)
    return answer['answer']

# Test it!
print("Q: What is RAG?")
print("A:", rag_answer("What is RAG?"))
print("\n---\n")
print("Q: Who is well positioned for AI roles?")
print("A:", rag_answer("Who is well positioned for AI roles?"))
