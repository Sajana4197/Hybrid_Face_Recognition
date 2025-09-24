#hypervector_generation.py

import numpy as np
import os
import json
from tqdm import tqdm

# Parameters
EMBEDDING_DIR = 'embeddings'
HYPERVECTOR_DIR = 'hypervectors'
DIM_ORIG = 512
DIM_HV = 10000

# Create fixed random projection matrix (shared for all encodings)
np.random.seed(42)
projection_matrix = np.random.randn(DIM_ORIG, DIM_HV)

def encode_embedding_to_hv(embedding):
    projected = np.dot(embedding, projection_matrix)  # shape: (10000,)
    hypervector = (projected > 0).astype(np.uint8)     # Binary HV (0 or 1)
    return hypervector

def process_embeddings():
    persons = [p for p in os.listdir(EMBEDDING_DIR) if os.path.isdir(os.path.join(EMBEDDING_DIR, p))]

    for person in tqdm(persons, desc="Encoding to hypervectors"):
        person_in_dir = os.path.join(EMBEDDING_DIR, person)
        person_out_dir = os.path.join(HYPERVECTOR_DIR, person)
        os.makedirs(person_out_dir, exist_ok=True)

        for file in os.listdir(person_in_dir):
            if file.endswith('.json'):
                path = os.path.join(person_in_dir, file)
                with open(path, 'r') as f:
                    data = json.load(f)
                    embedding = data['embedding'] if isinstance(data, dict) else data

                embedding = np.array(embedding)
                hv = encode_embedding_to_hv(embedding)

                # Save hypervector as binary list in JSON
                hv_path = os.path.join(person_out_dir, file.replace('.json', '_hv.json'))
                with open(hv_path, 'w') as f:
                    json.dump(hv.tolist(), f)

if __name__ == '__main__':
    process_embeddings()
    

      