import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from groq import Groq
from backend.config import GROQ_API_KEY, MODEL_NAME, PDF_FOLDER

# Initialize Groq
client = Groq(api_key=GROQ_API_KEY)

# Load PDFs
def load_documents():

    docs = []

    # Ensure PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        print(f"PDF folder not found: {PDF_FOLDER}")
        return docs

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            docs.extend(loader.load())

    return docs


# Build Vector DB
def build_vectorstore():

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = FastEmbedEmbeddings()
     
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# Initialize vectorstore
vectorstore = None

retriever = None

# RAG question answering
def ask_question(query):

    global vectorstore, retriever

    if vectorstore is None:
        vectorstore = build_vectorstore()
        retriever = vectorstore.as_retriever()

    docs = retriever.invoke(query)

    if not docs:
        return "No information found in the documents.", []

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an AI assistant answering questions ONLY from the provided context.

    Rules:
    - Answer clearly
    - Use bullet points
    - If information is not present say:
    "Information not found in the provided documents."

    Context:
    {context}

    Question:
    {query}
    """

    try:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = chat.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = "Server error occurred"

    return answer, docs
