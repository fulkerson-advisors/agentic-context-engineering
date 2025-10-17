"""
FAISS implementation of VectorIndex for semantic deduplication.
"""

import faiss
import numpy as np
import pickle
from typing import List, Optional, Dict
from pathlib import Path

from ace.core.interfaces import VectorIndex


class FAISSVectorIndex(VectorIndex):
    """
    FAISS-backed vector similarity search.
    
    Uses FAISS for efficient nearest neighbor search to detect
    semantically similar bullets for deduplication.
    """
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimensionality of vectors
            index_path: Optional path to load existing index
        """
        self.dimension = dimension
        self.index_path = index_path
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Map FAISS indices to bullet IDs
        self.id_map: Dict[int, str] = {}
        self.reverse_map: Dict[str, int] = {}
        self.next_idx = 0
        
        if index_path and Path(index_path).exists():
            self.load(index_path)
    
    def add_vectors(self, ids: List[str], vectors: List[List[float]]) -> None:
        """Add vectors to the index."""
        if not ids or not vectors:
            return
        
        # Convert to numpy and normalize for cosine similarity
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors_np = vectors_np / norms
        
        # Add to FAISS index
        self.index.add(vectors_np)
        
        # Update ID mappings
        for bullet_id in ids:
            self.id_map[self.next_idx] = bullet_id
            self.reverse_map[bullet_id] = self.next_idx
            self.next_idx += 1
    
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            threshold: Optional similarity threshold (0-1)
            
        Returns:
            List of (bullet_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_np = np.array([query_vector], dtype=np.float32)
        norm = np.linalg.norm(query_np)
        if norm > 0:
            query_np = query_np / norm
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            bullet_id = self.id_map.get(int(idx))
            if bullet_id is None:
                continue
            
            similarity = float(dist)  # Already cosine similarity due to normalization
            
            if threshold is None or similarity >= threshold:
                results.append((bullet_id, similarity))
        
        return results
    
    def remove_vector(self, vector_id: str) -> None:
        """
        Remove a vector from the index.
        
        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        For production, consider using IndexIDMap or periodic rebuilds.
        """
        if vector_id not in self.reverse_map:
            return
        
        # For now, we mark it as removed but don't rebuild
        # In production, you'd want to periodically rebuild the index
        idx = self.reverse_map[vector_id]
        del self.id_map[idx]
        del self.reverse_map[vector_id]
    
    def save(self, path: str) -> None:
        """Persist index to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path_obj))
        
        # Save ID mappings
        metadata = {
            'id_map': self.id_map,
            'reverse_map': self.reverse_map,
            'next_idx': self.next_idx,
            'dimension': self.dimension
        }
        
        with open(str(path_obj) + '.meta', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(path_obj))
        
        # Load ID mappings
        meta_path = str(path_obj) + '.meta'
        if Path(meta_path).exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.id_map = metadata['id_map']
            self.reverse_map = metadata['reverse_map']
            self.next_idx = metadata['next_idx']
            self.dimension = metadata['dimension']

