import faiss
import numpy as np
import json
from langchain_community.vectorstores import FAISS as LC_FAISS

def create_faiss_index(chunks, embeddings):
    """
    Create a FAISS index from a list of chunk dictionaries.
    Returns (index, embed_array).
    """
    texts = [chunk["text"] for chunk in chunks]
    embed_array = embeddings.embed_documents(texts)
    vector_size = np.array(embed_array).shape[1]
    index = faiss.IndexFlatL2(vector_size)
    index.add(np.array(embed_array).astype("float32"))
    return index, embed_array

def save_faiss_index(index, metadata, index_path='index.faiss', metadata_path='metadata.json'):
    """
    Save the FAISS index and its metadata to disk.
    """
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[INFO] FAISS index saved to {index_path} and metadata to {metadata_path}")

def load_faiss_index(index_path='index.faiss', metadata_path='metadata.json'):
    """
    Load a FAISS index and its metadata from disk.
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def add_chunks_to_index(index, existing_metadata, new_chunks, embeddings):
    """
    Add new chunks to an existing FAISS index and update metadata.
    """
    texts = [chunk["text"] for chunk in new_chunks]
    embed_array = embeddings.embed_documents(texts)
    index.add(np.array(embed_array).astype("float32"))
    existing_metadata.extend(new_chunks)
    return index, existing_metadata

def delete_chunk_from_index(index, metadata, chunk_id, embeddings):
    """
    Delete a chunk by ID by rebuilding the index.
    """
    updated_chunks = [item for item in metadata if item["id"] != chunk_id]
    new_texts = [item["text"] for item in updated_chunks]
    embed_array = embeddings.embed_documents(new_texts)
    vector_size = np.array(embed_array).shape[1]
    new_index = faiss.IndexFlatL2(vector_size)
    new_index.add(np.array(embed_array).astype("float32"))
    return new_index, updated_chunks
