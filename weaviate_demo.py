# **Weaviate**

# **Step 1: Install and Run Weaviate**
#    - You can run Weaviate using Docker:
#      ```bash
#      docker run -d -p 8080:8080 semitechnologies/weaviate
#      ```

# **Step 2: Install Weaviate Python Client**
#    - Run the following command to install the Weaviate client:
#      ```bash
#      pip install weaviate-client

# Purpose: This demo shows how to use Weaviate, a vector database, to power our bank's financial chatbot.
# We store sample transactions (e.g., credit card payments) with embeddings (numerical codes capturing meaning)
# and use vector search to find transactions similar to a customer's query, like "Customer used credit card for payment."
# The similarity scores help the chatbot retrieve the most relevant transactions for Retrieval-Augmented Generation (RAG).

# Import Weaviate to manage our vector database for the bank's chatbot.
import weaviate

# Import Weaviate classes (as wvc) for database setup tools.
import weaviate.classes as wvc

# Import SentenceTransformer to turn text into embeddings (numerical codes).
from sentence_transformers import SentenceTransformer

# Import warnings to manage Python warning messages.
import warnings

# Ignore ResourceWarning messages to keep demo output clean.
warnings.filterwarnings("ignore", category=ResourceWarning)

# Main code block: Runs only if this file is executed directly.
if __name__ == "__main__":
    # Step 1: Connect to Weaviate on localhost, port 8080.
    # 'client' lets us interact with Weaviate.
    client = weaviate.connect_to_local(host="localhost", port=8080)
    
    # Try block to handle errors gracefully.
    try:
        # Confirm connection to Weaviate for our chatbot.
        print("Connected to Weaviate! Ready to power our bank's financial chatbot! 💰")

        # Step 2: Load SentenceTransformer model to create 384D embeddings.
        # Embeddings capture the meaning of text (e.g., "credit card" and "debit card" are similar).
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Step 3: Name our Weaviate collection "Transaction" to store bank transactions.
        class_name = "Transaction"

        # Delete the collection if it exists to start fresh.
        if client.collections.exists(class_name):
            client.collections.delete(class_name)

        # Create a new collection in Weaviate for transactions.
        # - properties: Store "description" and "category" as text fields.
        # - vectorizer_config: We provide our own embeddings.
        # - vector_index_config: Use HNSW with cosine distance for similarity search (lower distance = more similar).
        client.collections.create(
            name=class_name,
            properties=[
                wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            )
        )

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

        # Pair each transaction with its embedding in sample_data.
        sample_data = [
            {"description": txn["description"], "category": txn["category"], "vector": emb}
            for txn, emb in zip(transactions, embeddings)
        ]

        # Step 5: Get the "Transaction" collection from Weaviate.
        transaction_collection = client.collections.get(class_name)

        # Add each transaction to Weaviate with its description, category, and embedding.
        for data in sample_data:
            transaction_collection.data.insert(
                properties={
                    "description": data["description"],
                    "category": data["category"]
                },
                vector=data["vector"]
            )

        # Confirm the transactions are stored and ready for search.
        print(f"Stored {len(sample_data)} financial transactions in Weaviate! Ready for vector search! 📊")

        # Step 6: Define the customer's question for the chatbot.
        query = "Customer used credit card for payment"

        # Turn the question into an embedding for searching.
        query_vector = model.encode([query]).tolist()[0]

        # Search for the 2 most similar transactions in Weaviate.
        # - near_vector: Use the query's embedding for similarity search.
        # - limit=2: Get the top 2 matches.
        # - return_properties: Retrieve description and category.
        # - return_metadata: Get the similarity score (cosine distance, lower is better).
        response = transaction_collection.query.near_vector(
            near_vector=query_vector,
            limit=2,
            return_properties=["description", "category"],
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )

        # Step 7: Print the search results header.
        print("\nVector Search Results for 'Customer used credit card for payment':")

        # Loop through the results and print each match's details.
        # The similarity score shows how close each transaction is to the query (0 is perfect, 2 is opposite).
        for item in response.objects:
            print(f"- Description: {item.properties['description']}")
            print(f"  Category: {item.properties['category']}")
            print(f"  Similarity Score (lower is better, cosine distance): {item.metadata.distance:.4f}\n")

    # Finally block: Close the Weaviate connection to clean up.
    finally:
        client.close()