from pdf_loader import load_and_split_pdf
from vector_store import create_vector_store
from chatbot import build_chatbot

def run_chatbot():
    pdf_path = "sample.pdf"  # replace with your PDF

    print("Loading PDF...")
    documents = load_and_split_pdf(pdf_path)

    print("Indexing documents in Pinecone...")
    vector_store = create_vector_store(documents)

    chatbot = build_chatbot(vector_store)

    print("\nPDF Chatbot Ready. Type 'exit' to quit.\n")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a valid question.")
            continue

        try:
            result = chatbot(query)

            if not result["result"]:
                print("Sorry, I couldn’t find relevant information.")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_chatbot()
