# File: milvus_demo.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

# Main code
if __name__ == "__main__":
    # Step 1: Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus!")

    # Step 2: Define and create a collection
    collection_name = "example_images"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),
        FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=100)
    ]
    schema = CollectionSchema(fields, "Image embeddings")
    collection = Collection(collection_name, schema)

    # Step 3: Create an index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    # Step 4: Generate sample image embeddings (simulating ResNet features)
    num_images = 5
    np.random.seed(42)
    image_embeddings = np.random.rand(num_images, 2048).tolist()  # 5 images, 2048D each
    image_names = [f"image_{i}.jpg" for i in range(num_images)]
    ids = list(range(num_images))

    # Step 5: Insert data into Milvus
    collection.insert([ids, image_embeddings, image_names])
    print(f"Saved {collection.num_entities} image embeddings to Milvus!")

    # Step 6: Query with a new vector
    query_vector = [np.random.rand(2048).tolist()]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=query_vector, anns_field="embedding", param=search_params, limit=3, output_fields=["image_name"])

    # Step 7: Display results
    print("\nQuery Results:")
    for result in results[0]:
        print(f"ID: {result.id}, Image: {result.entity.get('image_name')}, Distance: {result.distance:.4f}")