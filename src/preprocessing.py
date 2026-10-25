
from langchain_text_splitters import RecursiveCharacterTextSplitter



def split_documents(documents):

    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Divide documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    # Load PDF documents
    documents = load_pdf_documents()

    print("Total pages:", len(documents))

    # Split the documents
    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))

    # Display the first chunk
    if chunks:

        print("\nFirst Chunk:\n")
        print(chunks[0].page_content)

        print("\nChunk Metadata:")
        print(chunks[0].metadata)
