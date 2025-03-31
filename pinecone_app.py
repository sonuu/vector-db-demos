# **Pinecone**

# **Step 1: Sign up and Get API Key**
#    - Visit [Pinecone.io](https://www.pinecone.io) and sign up for an account.
#    - After signing up, locate your API key in the dashboard.

# **Step 2: Install Pinecone Client Library**
#    - Run the following command to install the Pinecone client:
#      ```bash
#      pip install pinecone-client

# Purpose: This demo shows how to use Pinecone, a cloud-based vector database, to power our bank's financial recommendation system.
# We store sample transactions (e.g., credit card payments) with embeddings and metadata (e.g., category)
# and use vector search to recommend transactions similar to a customer's query, like "Customer used credit card for payment."
# The results help the system suggest related financial activities for Retrieval-Augmented Generation (RAG).
# Pinecone excels with its cloud-based scalability, managed service, and metadata support—perfect for our production needs!

# Import Pinecone for cloud-based vector database operations.
from pinecone import Pinecone, ServerlessSpec

# Import SentenceTransformer to turn text into embeddings (numerical codes).
from sentence_transformers import SentenceTransformer

# Main code block: Runs only if this file is executed directly.
if __name__ == "__main__":
    # Step 1: Initialize Pinecone with your API key.
    # Pinecone's managed service means we don't worry about infrastructure—it's all in the cloud!
    pc = Pinecone(api_key="pcsk_746V7J_FYved1BK5rtcjaNmxjNtmYb9kXFhbSJ6ikrTyRfzty1FRwyTv6z5kJsky81yAkY")

    # Step 2: Define the index name and dimensions (384 for our embeddings).
    index_name = "financial-transactions"
    dimension = 384

    # Delete the index if it exists to start fresh.
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create a new serverless index in Pinecone.
    # - dimension: 384 for our embeddings.
    # - metric: Cosine similarity for semantic search.
    # Pinecone's scalability shines here—it can handle millions of vectors effortlessly!
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Connect to the index.
    index = pc.Index(index_name)

    # Step 3: Load SentenceTransformer model to create 384D embeddings.
    # Embeddings capture the meaning of text (e.g., "credit card" and "debit card" are similar).
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Step 4: Prepare 3 sample transactions for our bank.
    transactions = [
        {"description": "Customer paid $500 using credit card for online shopping on 2025-03-01.", "category": "Credit Card"},
        {"description": "Customer transferred $1000 to savings account on 2025-03-02.", "category": "Savings"},
        {"description": "Customer used debit card to withdraw $200 from ATM on 2025-03-03.", "category": "Debit Card"},
    ]

    # Extract descriptions to generate embeddings.
    descriptions = [txn["description"] for txn in transactions]

    # Generate embeddings for the descriptions (384 numbers per description).
    embeddings = model.encode(descriptions).tolist()

    # Prepare data for Pinecone: each transaction gets an ID, embedding, and metadata.
    # Pinecone's metadata support lets us store the category alongside the vector—great for rich queries!
    vectors = [
        (f"txn_{i}", emb, {"category": txn["category"], "description": txn["description"]})
        for i, (emb, txn) in enumerate(zip(embeddings, transactions))
    ]

    # Step 5: Upsert the vectors into Pinecone.
    # Pinecone's managed service handles indexing and storage automatically—zero hassle!
    index.upsert(vectors=vectors)

    # Confirm the transactions are stored and ready for search.
    print(f"Stored {len(transactions)} financial transactions in Pinecone! Ready for vector search! ☁️")

    # Step 6: Define the customer's question for the recommendation system.
    query = "Customer used credit card for payment"

    # Turn the question into an embedding for searching.
    query_vector = model.encode([query]).tolist()[0]

    # Step 7: Search for the 2 most similar transactions in Pinecone.
    # - top_k=2: Get the top 2 matches.
    # - include_metadata=True: Retrieve the category and description.
    # Pinecone's cloud scalability ensures this works even with massive datasets!
    response = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )

    # Step 8: Print the search results header.
    print("\nVector Search Results for 'Customer used credit card for payment':")

    # Loop through the results and print each match's details.
    # - score: The similarity score (cosine similarity, higher is better, 1 is perfect).
    # Pinecone's metadata support makes it easy to retrieve the category and description!
    for match in response['matches']:
        print(f"- Description: {match['metadata']['description']}")
        print(f"  Category: {match['metadata']['category']}")
        print(f"  Similarity Score (higher is better, cosine similarity): {match['score']:.4f}\n")