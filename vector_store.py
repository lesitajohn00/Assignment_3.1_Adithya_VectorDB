import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def create_vector_store(documents):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Pinecone error: {str(e)}")
