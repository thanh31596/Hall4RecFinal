import numpy as np
from typing import Tuple
from config import HallAgentConfig

class CollaborativeFiltering:
    """Matrix Factorization for Collaborative Filtering"""
    
    def __init__(self, config: HallAgentConfig):
        self.config = config
        
    def fit(self, interaction_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Learn user and item embeddings via matrix factorization
        """
        n_users, n_items = interaction_matrix.shape
        
        # Initialize embeddings randomly
        np.random.seed(self.config.random_seed)
        user_embeddings = np.random.normal(0, 0.1, (n_users, self.config.latent_dim))
        item_embeddings = np.random.normal(0, 0.1, (n_items, self.config.latent_dim))
        
        # Get observed interactions
        observed_mask = interaction_matrix > 0
        n_observed = np.sum(observed_mask)
        
        print(f"Training CF on {n_observed} observed interactions...")
        
        # SGD optimization
        for iteration in range(self.config.cf_max_iterations):
            total_loss = 0
            n_updates = 0
            
            # Iterate through observed interactions
            user_indices, item_indices = np.where(observed_mask)
            
            for idx in range(len(user_indices)):
                i, j = user_indices[idx], item_indices[idx]
                
                # Prediction
                pred = np.dot(user_embeddings[i], item_embeddings[j])
                error = interaction_matrix[i, j] - pred
                
                # Gradients
                u_grad = error * item_embeddings[j] - self.config.cf_reg_u * user_embeddings[i]
                v_grad = error * user_embeddings[i] - self.config.cf_reg_v * item_embeddings[j]
                
                # Updates
                user_embeddings[i] += self.config.cf_learning_rate * u_grad
                item_embeddings[j] += self.config.cf_learning_rate * v_grad
                
                total_loss += error ** 2
                n_updates += 1
            
            # Add regularization to loss
            reg_loss = (self.config.cf_reg_u * np.sum(user_embeddings ** 2) + 
                       self.config.cf_reg_v * np.sum(item_embeddings ** 2))
            total_loss += reg_loss
            
            if iteration % 10 == 0:
                avg_loss = total_loss / n_updates if n_updates > 0 else 0
                print(f"CF Iteration {iteration}: RMSE = {np.sqrt(avg_loss):.4f}")
        
        print("CF training completed!")
        return user_embeddings, item_embeddings
    
    def predict(self, user_embeddings: np.ndarray, item_embeddings: np.ndarray,
                user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        return np.dot(user_embeddings[user_id], item_embeddings[item_id])