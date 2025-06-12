import numpy as np
from typing import Tuple
from config import HallAgentConfig

class TransferMatrixLearner:
    """Learn transfer matrices and projection matrices"""
    
    def __init__(self, config: HallAgentConfig):
        self.config = config
        
    def learn_transfer_matrix(self, user_vectors: np.ndarray, 
                            item_features: np.ndarray,
                            interaction_matrix: np.ndarray,
                            item_biases: np.ndarray) -> np.ndarray:
        """Learn transfer matrix A using closed-form solution"""
        print("Learning transfer matrix A...")
        
        k, f = user_vectors.shape[1], item_features.shape[1]
        
        # Initialize matrices for accumulation
        numerator = np.zeros((k, f))
        denominator = np.zeros((f, f))
        
        # Accumulate over observed interactions
        observed_mask = interaction_matrix > 0
        n_observed = 0
        
        for i in range(interaction_matrix.shape[0]):
            for j in range(interaction_matrix.shape[1]):
                if observed_mask[i, j]:
                    rating = interaction_matrix[i, j]
                    bias = item_biases[j] if j < len(item_biases) else 0
                    
                    # Accumulate for closed-form solution
                    numerator += (rating - bias) * np.outer(user_vectors[i], item_features[j])
                    denominator += np.outer(item_features[j], item_features[j])
                    n_observed += 1
        
        print(f"Processed {n_observed} interactions for transfer matrix learning")
        
        # Add regularization
        denominator += self.config.transfer_reg * np.eye(f)
        
        # Solve: A = numerator @ inv(denominator)
        try:
            transfer_matrix = numerator @ np.linalg.inv(denominator)
        except np.linalg.LinAlgError:
            print("Warning: Using pseudo-inverse due to singular matrix")
            transfer_matrix = numerator @ np.linalg.pinv(denominator)
        
        print(f"Transfer matrix A shape: {transfer_matrix.shape}")
        return transfer_matrix
    
    def construct_projection_matrix(self, user_vectors: np.ndarray) -> np.ndarray:
        """Construct projection matrix B using PCA"""
        print("Constructing projection matrix B...")
        
        # Compute user covariance matrix
        user_mean = np.mean(user_vectors, axis=0)
        centered_users = user_vectors - user_mean
        cov_matrix = (centered_users.T @ centered_users) / (user_vectors.shape[0] - 1)
        
        # Perform PCA
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        sorted_eigenvals = eigenvals[idx]
        sorted_eigenvecs = eigenvecs[:, idx]
        
        # Take top-l eigenvectors - FIXED: Removed .T to get correct dimensions
        # This gives us B of shape (latent_dim, low_rank_dim) = (k, ℓ)
        # which is needed for the computation u^T_i B θ_j in equation (12)
        B = sorted_eigenvecs[:, :self.config.low_rank_dim]
        
        # Calculate variance preserved
        total_variance = np.sum(eigenvals)
        preserved_variance = np.sum(sorted_eigenvals[:self.config.low_rank_dim])
        variance_ratio = preserved_variance / total_variance
        
        print(f"Projection matrix B shape: {B.shape}")
        print(f"Expected shape: ({user_vectors.shape[1]}, {self.config.low_rank_dim})")
        print(f"Variance preserved: {variance_ratio:.3f}")
        
        return B
    
    def learn_context_weights(self, user_vectors: np.ndarray,
                            item_features: np.ndarray,
                            interaction_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Learn context weights and item biases"""
        print("Learning context weights and item biases...")
        
        n_items = interaction_matrix.shape[1]
        
        # Initialize context weights (placeholder for future context features)
        context_dim = 10  # Adjust based on your context features
        context_weights = np.random.normal(0, 0.01, context_dim)
        
        # Initialize item biases as mean ratings
        item_biases = np.zeros(n_items)
        for j in range(n_items):
            item_ratings = interaction_matrix[:, j]
            observed_ratings = item_ratings[item_ratings > 0]
            if len(observed_ratings) > 0:
                item_biases[j] = np.mean(observed_ratings)
            else:
                item_biases[j] = 3.0  # Default to middle rating
        
        global_mean = np.mean(item_biases)
        print(f"Item biases - Mean: {global_mean:.3f}, Std: {np.std(item_biases):.3f}")
        
        return context_weights, item_biases