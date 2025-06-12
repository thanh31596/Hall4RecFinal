import numpy as np
from typing import Tuple
from config import HallAgentConfig

class AttentionFusion:
    """Attention-based fusion of CF and personality vectors"""
    
    def __init__(self, config: HallAgentConfig):
        self.config = config
        self.W_cf = None
        self.W_gen = None
        self.b_cf = None
        self.b_gen = None
        
    def fuse_representations(self, cf_vectors: np.ndarray, 
                           personality_vectors: np.ndarray,
                           interaction_matrix: np.ndarray) -> np.ndarray:
        """Fuse CF and personality vectors using attention mechanism"""
        n_users, cf_dim = cf_vectors.shape
        _, gen_dim = personality_vectors.shape
        target_dim = self.config.latent_dim
        
        print("Fusing CF and personality representations...")
        
        # Initialize projection matrices
        np.random.seed(self.config.random_seed)
        self.W_cf = np.random.normal(0, 0.1, (target_dim, cf_dim))
        self.W_gen = np.random.normal(0, 0.1, (target_dim, gen_dim))
        self.b_cf = np.zeros(target_dim)
        self.b_gen = np.zeros(target_dim)
        
        # Learn projections via simple optimization
        self._learn_projections(cf_vectors, personality_vectors, interaction_matrix)
        
        # Apply learned projections
        h_cf = cf_vectors @ self.W_cf.T + self.b_cf
        h_gen = personality_vectors @ self.W_gen.T + self.b_gen
        
        # Compute attention weights
        attention_scores = np.sum(h_cf * h_gen, axis=1)
        alpha = self._sigmoid(attention_scores)
        
        print(f"Attention weights - Mean: {np.mean(alpha):.3f}, Std: {np.std(alpha):.3f}")
        
        # Fuse representations
        fused_vectors = np.zeros((n_users, target_dim))
        for i in range(n_users):
            fused_vectors[i] = alpha[i] * h_gen[i] + (1 - alpha[i]) * h_cf[i]
        
        print("Fusion completed!")
        return fused_vectors
    
    def _learn_projections(self, cf_vectors: np.ndarray, 
                          personality_vectors: np.ndarray,
                          interaction_matrix: np.ndarray):
        """Learn projection matrices via simple optimization"""
        # Simplified learning - project to same dimension while preserving information
        # In practice, you would use gradient descent with reconstruction loss
        
        # For CF vectors - simple linear projection to target dimension
        if cf_vectors.shape[1] != self.config.latent_dim:
            U, _, Vt = np.linalg.svd(cf_vectors, full_matrices=False)
            self.W_cf = Vt[:self.config.latent_dim, :]
        else:
            self.W_cf = np.eye(self.config.latent_dim)
        
        # For personality vectors - PCA projection
        personality_mean = np.mean(personality_vectors, axis=0)
        centered_personality = personality_vectors - personality_mean
        
        cov_matrix = (centered_personality.T @ centered_personality) / (personality_vectors.shape[0] - 1)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending) and take top components
        idx = np.argsort(eigenvals)[::-1]
        self.W_gen = eigenvecs[:, idx[:self.config.latent_dim]].T
        self.b_gen = -self.W_gen @ personality_mean
    
    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))