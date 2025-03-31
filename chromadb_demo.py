# **ChromaDB**

# **Step 1: Install ChromaDB**
#    - Run the following command to install ChromaDB:
#      ```bash
#      pip install chromadb
#      ```

# **Step 2: Create and Query Embedding Collection**
#    - Use the provided code to initialize a ChromaDB client, create an embedding collection, insert data, and perform a search.
# Purpose: This demo shows how to use ChromaDB, a lightweight vector database, to power our bank's financial chatbot.
# We store sample transactions (e.g., credit card payments) with embeddings and their descriptions
# and use vector search to find transactions similar to a customer's query, like "Customer used credit card for payment."
# The results help the chatbot retrieve the most relevant transactions for Retrieval-Augmented Generation (RAG).
# ChromaDB shines with its simplicity, persistence, and efficiency for text embeddings—perfect for quick prototyping!

# Import ChromaDB for a lightweight vector database.
import chromadb

# Import SentenceTransformer to turn text into embeddings (numerical codes).
from sentence_transformers import SentenceTransformer

# Main code block: Runs only if this file is executed directly.
if __name__ == "__main__":
    # Step 1: Initialize ChromaDB with persistence.
    # - path="./chromadb_data": Save the database to disk for reuse.
    # ChromaDB's persistence feature ensures we don't lose our data—great for real-world use!
    client = chromadb.PersistentClient(path="./chromadb_data")

    # Create or get a collection named "transactions".
    collection = client.get_or_create_collection(name="transactions")

    # Step 2: Load SentenceTransformer model to create 384D embeddings.
    # Embeddings capture the meaning of text (e.g., "credit card" and "debit card" are similar).
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Step 3: Prepare 3 sample transactions for our bank.
    transactions = [
        {"description": "Customer paid $500 using credit card for online shopping on 2025-03-01.", "category": "Credit Card"},
        {"description": "Customer transferred $1000 to savings account on 2025-03-02.", "category": "Savings"},
        {"description": "Customer used debit card to withdraw $200 from ATM on 2025-03-03.", "category": "Debit Card"},
    ]

    # Extract descriptions to generate embeddings.
    descriptions = [txn["description"] for txn in transactions]

    # Generate embeddings for the descriptions (384 numbers per description).
    embeddings = model.encode(descriptions).tolist()

    # Prepare IDs for each transaction.
    ids = [f"txn_{i}" for i in range(len(transactions))]

    # Step 4: Add the transactions to ChromaDB.
    # - documents: Store the raw descriptions.
    # - embeddings: Store the 384D embeddings.
    # - ids: Assign unique IDs to each transaction.
    # ChromaDB's efficiency for text embeddings makes this step a breeze!
    collection.add(
        documents=descriptions,
        embeddings=embeddings,
        ids=ids,
        metadatas=transactions  # Store the full transaction data as metadata
    )

    # Confirm the transactions are stored and ready for search.
    print(f"Stored {len(transactions)} financial transactions in ChromaDB! Ready for vector search! 🖥️")

    # Step 5: Define the customer's question for the chatbot.
    query = "Customer used credit card for payment"

    # Turn the question into an embedding for searching.
    query_vector = model.encode([query]).tolist()[0]

    # Step 6: Search for the 2 most similar transactions in ChromaDB.
    # - query_embeddings: Use the query's embedding for similarity search.
    # - n_results=2: Get the top 2 matches.
    # ChromaDB's simplicity makes vector search straightforward and fast!
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    # Step 7: Print the search results header.
    print("\nVector Search Results for 'Customer used credit card for payment':")

    # Loop through the results and print each match's details.
    # - documents: The transaction descriptions.
    # - metadatas: The full transaction data (including category).
    # - distances: The similarity scores (cosine distance, lower is better).
    # ChromaDB's lightweight design ensures this runs smoothly on our local setup!
    for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        print(f"- Description: {doc}")
        print(f"  Category: {metadata['category']}")
        print(f"  Similarity Score (lower is better, cosine distance): {distance:.4f}\n")

# No need to close ChromaDB—the PersistentClient saves the data to disk automatically.