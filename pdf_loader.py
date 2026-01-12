from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("PDF is empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        split_docs = text_splitter.split_documents(documents)

        # Add metadata
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["char_count"] = len(doc.page_content)

        return split_docs

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")
