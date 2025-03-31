# **Faiss**

# **Step 1: Install Faiss**
#    - Use the following command to install Faiss:
#      ```bash
#      pip install faiss-cpu
#      ```


# Purpose: This demo shows how to use Faiss, a high-performance library for similarity search, to power our bank's financial chatbot.
# We store sample transactions (e.g., credit card payments) with embeddings (numerical codes capturing meaning)
# and use vector search to find transactions similar to a customer's query, like "Customer used credit card for payment."
# The similarity scores help the chatbot retrieve the most relevant transactions for Retrieval-Augmented Generation (RAG).
# Faiss shines with its lightning-fast search and flexible indexing, making it perfect for our high-dimensional data needs!

# Import Faiss for high-performance similarity search—Faiss is the speed king for vector search!
import faiss

# Import NumPy for numerical operations on embeddings.
import numpy as np

# Import SentenceTransformer to turn text into embeddings (numerical codes).
from sentence_transformers import SentenceTransformer

# Main code block: Runs only if this file is executed directly.
if __name__ == "__main__":
    # Step 1: Load SentenceTransformer model to create 384D embeddings.
    # Embeddings capture the meaning of text (e.g., "credit card" and "debit card" are similar).
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Step 2: Prepare 3 sample transactions for our bank.
    transactions = [
        {"description": "Customer paid $500 using credit card for online shopping on 2025-03-01.", "category": "Credit Card"},
        {"description": "Customer transferred $1000 to savings account on 2025-03-02.", "category": "Savings"},
        {"description": "Customer used debit card to withdraw $200 from ATM on 2025-03-03.", "category": "Debit Card"},
    ]

    # Extract descriptions to generate embeddings.
    descriptions = [txn["description"] for txn in transactions]

    # Generate embeddings for the descriptions (384 numbers per description).
    embeddings = model.encode(descriptions)

    # Convert embeddings to a NumPy array for Faiss (shape: 3 transactions x 384 dimensions).
    embeddings = np.array(embeddings).astype('float32')

    # Step 3: Create a Faiss index using LSH (Locality-Sensitive Hashing).
    # - 384: Number of dimensions in our embeddings.
    # - LSH groups similar vectors into buckets for fast search—Faiss's flexibility shines here!
    dimension = 384
    index = faiss.IndexLSH(dimension, 128)  # 128 bits for hashing

    # Add the embeddings to the Faiss index.
    # Faiss's speed makes this step lightning-fast, even for large datasets!
    index.add(embeddings)

    # Step 4: Check the data stored in Faiss.
    # - index.ntotal: Number of vectors in the index.
    # Since Faiss doesn't store metadata, we reference the original transactions list.
    print(f"\nChecking Data in Faiss:")
    print(f"Total number of vectors stored: {index.ntotal}")
    print("Stored Transactions:")
    for i in range(index.ntotal):
        print(f"- Transaction {i}:")
        print(f"  Description: {transactions[i]['description']}")
        print(f"  Category: {transactions[i]['category']}")
        # Note: We can't directly retrieve embeddings with LSH, but we can with other index types like IndexFlatL2.

    # Confirm the transactions are stored and ready for search.
    print(f"\nStored {len(transactions)} financial transactions in Faiss! Ready for vector search! 📈")

    # Step 5: Define the customer's question for the chatbot.
    query = "Customer used credit card for payment"

    # Turn the question into an embedding for searching.
    query_vector = model.encode([query])

    # Convert the query embedding to a NumPy array for Faiss.
    query_vector = np.array(query_vector).astype('float32')

    # Step 6: Search for the 2 most similar transactions in Faiss.
    # - k=2: Get the top 2 matches.
    # - Faiss's lightning-fast search retrieves results in milliseconds!
    k = 2
    distances, indices = index.search(query_vector, k)

    # Step 7: Print the search results header.
    print("\nVector Search Results for 'Customer used credit card for payment':")

    # Loop through the results and print each match's details.
    # - indices: The positions of the matching transactions in our list.
    # - distances: The similarity scores (LSH uses Hamming distance, lower is better).
    # Faiss's scalability ensures this works even with millions of transactions!
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"- Match {i+1}:")
        print(f"  Description: {transactions[idx]['description']}")
        print(f"  Category: {transactions[idx]['category']}")
        print(f"  Similarity Score (lower is better, Hamming distance): {distance:.2f}\n")

# No need to close Faiss—it's a lightweight library with no persistent connection.