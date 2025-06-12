import numpy as np
from typing import Dict, Optional, Tuple
from config import HallAgentConfig

class OnlineLearner:
    """Online learning for adaptation parameters"""
    
    def __init__(self, config: HallAgentConfig):
        self.config = config
        self.online_factors = {}  # item_id -> theta_j
        
    def initialize_item_factors(self, n_items: int):
        """Initialize online factors for all items"""
        for item_id in range(n_items):
            self.online_factors[item_id] = np.zeros(self.config.low_rank_dim)
    
    def update_parameters(self, interaction: Tuple[int, int, float],
                         user_vectors: np.ndarray,
                         item_features: np.ndarray,
                         transfer_matrix_A: np.ndarray,
                         projection_matrix_B: np.ndarray,
                         context_weights: np.ndarray,
                         item_biases: np.ndarray,
                         context_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Update online parameters based on new interaction"""
        user_id, item_id, rating = interaction
        
        # Predict current rating
        predicted_rating = self._predict_rating(
            user_id, item_id, user_vectors, item_features,
            transfer_matrix_A, projection_matrix_B,
            context_weights, item_biases, context_features
        )
        
        # Calculate error
        error = rating - predicted_rating
        
        # Update online factors theta_j
        if item_id not in self.online_factors:
            self.online_factors[item_id] = np.zeros(self.config.low_rank_dim)
        
        user_vec = user_vectors[user_id]
        gradient_theta = error * (projection_matrix_B.T @ user_vec) - \
                        self.config.online_reg_theta * self.online_factors[item_id]
        
        self.online_factors[item_id] += self.config.online_lr_theta * gradient_theta
        
        # Update item bias
        gradient_bias = error - self.config.online_reg_theta * item_biases[item_id]
        item_biases[item_id] += self.config.bias_lr * gradient_bias
        
        # Update context weights if context features provided
        if context_features is not None:
            gradient_context = error * context_features - \
                             self.config.online_reg_theta * context_weights
            context_weights += self.config.context_lr * gradient_context
        
        return {
            'error': error,
            'predicted_rating': predicted_rating,
            'actual_rating': rating
        }
    
    def _predict_rating(self, user_id: int, item_id: int,
                       user_vectors: np.ndarray,
                       item_features: np.ndarray,
                       transfer_matrix_A: np.ndarray,
                       projection_matrix_B: np.ndarray,
                       context_weights: np.ndarray,
                       item_biases: np.ndarray,
                       context_features: Optional[np.ndarray] = None) -> float:
        """Predict rating for user-item pair"""
        user_vec = user_vectors[user_id]
        
        # Content-based score
        content_score = user_vec @ transfer_matrix_A @ item_features[item_id]
        
        # Online adaptation score
        online_score = 0.0
        if item_id in self.online_factors:
            online_score = user_vec @ projection_matrix_B @ self.online_factors[item_id]
        
        # Context score
        context_score = 0.0
        if context_features is not None:
            context_score = context_weights @ context_features
        
        # Bias
        bias = item_biases[item_id]
        
        # Combined prediction
        linear_score = content_score + online_score + context_score + bias
        return 1 + 4 * self._sigmoid(linear_score)
    
    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))