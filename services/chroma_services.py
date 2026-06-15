import chromadb
import uuid

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


def clear_collection():

    global collection

    try:
        client.delete_collection(
            "documents"
        )
    except:
        pass

    collection = client.get_or_create_collection(
        name="documents"
    )


def store_chunks(
    chunks,
    embeddings
):

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )


def search_chunks(
    query_embedding,
    top_k=5
):

    return collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )