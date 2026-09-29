import os
import json
import numpy as np
import logging
from numpy.linalg import norm

logger = logging.getLogger("samvaad.enrollment")

class EnrollmentEngine:
    def __init__(self, vocab_path="models/personal_vocab.json"):
        self.vocab_path = vocab_path
        self.personal_vocab = {}
        self.load_vocab()
        
    def load_vocab(self):
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r") as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays
                    self.personal_vocab = {
                        word: np.array(emb, dtype=np.float32) 
                        for word, emb in data.items()
                    }
                logger.info(f"Loaded {len(self.personal_vocab)} personal signs.")
            except Exception as e:
                logger.error(f"Failed to load personal vocab: {e}")
        else:
            os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
            self.personal_vocab = {}
            
    def save_vocab(self):
        try:
            # Convert numpy arrays to lists for JSON serialization
            data = {
                word: emb.tolist() 
                for word, emb in self.personal_vocab.items()
            }
            with open(self.vocab_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save personal vocab: {e}")

    def enroll_sign(self, word: str, embeddings: list):
        """
        Takes a list of 5 embeddings (128-d vectors) for a new word,
        averages them into a prototype, and saves it.
        """
        if len(embeddings) < 3:
            raise ValueError("Need at least 3 examples to enroll a stable sign.")
            
        # Convert to numpy array: shape (N, 128)
        emb_matrix = np.array(embeddings, dtype=np.float32)
        
        # Calculate the mean prototype vector: shape (128,)
        prototype = np.mean(emb_matrix, axis=0)
        
        # Normalize the prototype vector to unit length for easier cosine similarity
        norm_val = norm(prototype)
        if norm_val > 1e-6:
            prototype = prototype / norm_val
            
        self.personal_vocab[word.upper()] = prototype
        self.save_vocab()
        logger.info(f"Successfully enrolled custom sign: {word.upper()}")
        
        return True

def get_cosine_similarity(vecA: np.ndarray, vecB: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    n_a = norm(vecA)
    n_b = norm(vecB)
    if n_a < 1e-6 or n_b < 1e-6:
        return 0.0
    return np.dot(vecA, vecB) / (n_a * n_b)
