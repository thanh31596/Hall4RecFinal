import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
import time
import random

from config import HallAgentConfig
from collaborative_filtering import CollaborativeFiltering
from personality_generator import PersonalityVectorGenerator
from attention_fusion import AttentionFusion
from transfer_learner import TransferMatrixLearner
from online_learner import OnlineLearner
from hallucination_detector import HallucinationDetector
from utils import safe_llm_invoke

class HallAgent4Rec:
    """Main HallAgent4Rec Framework"""
    
    def __init__(self, config: HallAgentConfig, llm, embeddings_model):
        self.config = config
        self.llm = llm
        self.embeddings_model = embeddings_model
        
        # Initialize components
        self.cf_module = CollaborativeFiltering(config)
        self.personality_generator = PersonalityVectorGenerator(llm, embeddings_model)
        self.attention_fusion = AttentionFusion(config)
        self.transfer_learner = TransferMatrixLearner(config)
        self.online_learner = OnlineLearner(config)
        self.hallucination_detector = HallucinationDetector(config, llm)
        
        # Model parameters
        self.user_vectors = None
        self.item_vectors = None
        self.transfer_matrix_A = None
        self.projection_matrix_B = None
        self.context_weights = None
        self.item_biases = None
        self.is_trained = False
        
    def offline_training(self, interaction_matrix: np.ndarray, 
                        item_features: np.ndarray,
                        user_demographics: Dict[int, Dict],
                        item_metadata: Dict[int, Dict]) -> Dict[str, Any]:
        """Algorithm 1: Offline Training Phase"""
        print("=" * 50)
        print("Starting HallAgent4Rec Offline Training")
        print("=" * 50)
        
        # Step 1: Learn Collaborative Filtering Representations
        print("\n1. Learning Collaborative Filtering representations...")
        cf_user_vectors, self.item_vectors = self.cf_module.fit(interaction_matrix)
        
        # Step 2: Generate Personality Vectors (with rate limiting)
        print("\n2. Generating personality vectors with rate limiting...")
        personality_vectors = self.personality_generator.generate_personality_vectors(
            user_demographics, interaction_matrix, item_metadata
        )
        
        # Step 3: Attention-Based Fusion
        print("\n3. Fusing representations with attention mechanism...")
        self.user_vectors = self.attention_fusion.fuse_representations(
            cf_user_vectors, personality_vectors, interaction_matrix
        )
        
        # Step 4: Learn Context Weights and Biases (before transfer matrix)
        print("\n4. Learning context weights and item biases...")
        self.context_weights, self.item_biases = self.transfer_learner.learn_context_weights(
            self.user_vectors, item_features, interaction_matrix
        )
        
        # Step 5: Learn Transfer Matrix A
        print("\n5. Learning transfer matrix...")
        self.transfer_matrix_A = self.transfer_learner.learn_transfer_matrix(
            self.user_vectors, item_features, interaction_matrix, self.item_biases
        )
        
        # Step 6: Construct Projection Matrix B
        print("\n6. Constructing projection matrix...")
        self.projection_matrix_B = self.transfer_learner.construct_projection_matrix(
            self.user_vectors
        )
        
        # Initialize online factors
        print("\n7. Initializing online factors...")
        self.online_learner.initialize_item_factors(interaction_matrix.shape[1])
        
        self.is_trained = True
        print("\n" + "=" * 50)
        print("Offline training completed successfully!")
        print("=" * 50)
        
        return {
            'user_vectors': self.user_vectors,
            'transfer_matrix_A': self.transfer_matrix_A,
            'projection_matrix_B': self.projection_matrix_B,
            'context_weights': self.context_weights,
            'item_biases': self.item_biases
        }
    
    def online_recommendation(self, user_id: int, candidate_items: List[int],
                            item_features: np.ndarray,
                            item_metadata: Dict[int, Dict],
                            context_features: Optional[np.ndarray] = None,
                            user_profile: str = "",
                            new_interaction: Optional[Tuple[int, int, float]] = None) -> List[int]:
        """Algorithm 2: Online Recommendation and Hallucination Mitigation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Phase 1: Score Candidate Items
        scored_items = self._score_candidate_items(
            user_id, candidate_items, item_features, context_features
        )
        
        # Phase 2: LLM Recommendation Generation (with rate limiting)
        llm_recommendations = self._generate_llm_recommendations_safe(
            user_profile, scored_items, item_features, item_metadata
        )
        
        # Phase 3: Hallucination Detection and Mitigation
        final_recommendations = self.hallucination_detector.detect_and_mitigate(
            llm_recommendations, candidate_items, user_id, item_features,
            self.user_vectors, self.transfer_matrix_A, self.projection_matrix_B,
            self.online_learner.online_factors, self.context_weights, 
            self.item_biases, item_metadata
        )
        
        # Phase 4: Online Parameter Updates
        if new_interaction:
            self.online_learner.update_parameters(
                new_interaction, self.user_vectors, item_features,
                self.transfer_matrix_A, self.projection_matrix_B,
                self.context_weights, self.item_biases, context_features
            )
        
        return final_recommendations
    
    def _score_candidate_items(self, user_id: int, candidate_items: List[int],
                              item_features: np.ndarray, 
                              context_features: Optional[np.ndarray]) -> List[Tuple[int, float]]:
        """Score candidate items using hybrid bilinear function"""
        scores = []
        user_vec = self.user_vectors[user_id]
        
        for item_id in candidate_items:
            # Content-based component
            content_score = user_vec @ self.transfer_matrix_A @ item_features[item_id]
            
            # Online adaptation component
            online_score = 0.0
            if item_id in self.online_learner.online_factors:
                online_score = user_vec @ self.projection_matrix_B @ self.online_learner.online_factors[item_id]
            
            # Context component
            context_score = 0.0
            if context_features is not None:
                context_score = self.context_weights @ context_features
            
            # Bias
            bias_score = self.item_biases[item_id]
            
            # Combined score
            linear_score = content_score + online_score + context_score + bias_score
            final_score = 1 + 4 * self._sigmoid(linear_score)
            
            scores.append((item_id, final_score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _generate_llm_recommendations_safe(self, user_profile: str, 
                                         scored_items: List[Tuple[int, float]],
                                         item_features: np.ndarray,
                                         item_metadata: Dict[int, Dict]) -> List[str]:
        """Generate recommendations using LLM. If LLM fails or returns nothing, report and terminate."""
        import sys
        # Get top-K scored items for LLM input
        top_items = scored_items[:20]
        
        # Create item descriptions for LLM prompt
        item_descriptions = []
        for item_id, score in top_items:
            if item_id in item_metadata:
                title = item_metadata[item_id]['title']
                genres = ', '.join(item_metadata[item_id]['genres'])
                year = item_metadata[item_id]['release_year']
                description = f"Movie ID {item_id + 1}: {title} ({year}) - Genres: {genres}"
                item_descriptions.append(description)
        
        # Create LLM prompt
        prompt = f"""You are a movie recommendation system for a user with the following profile: {user_profile}

        Based on the user's profile and preferences, here are some relevant movies from our catalog:

        {chr(10).join(item_descriptions)}

        Please recommend exactly 10 movies from the list above that would be most relevant for this user.
        For each recommendation, provide the Movie ID and a brief explanation.

        IMPORTANT: You must ONLY recommend movies from the provided list. Use the exact Movie ID format shown above.

        Format your response as:
        Movie ID X: [Brief explanation]
        Movie ID Y: [Brief explanation]
        ..."""

        try:
            response_content = safe_llm_invoke(self.llm, prompt)
            recommendations = self._parse_llm_response(response_content)
            print("Got the movie")
            print("="*60)
            print("Response content: ",response_content)
            print("="*60)
            if not recommendations:
                print("LLM returned empty recommendations. Terminating program.")
                sys.exit(1)
            return recommendations
        except Exception as e:
            print(f"Error generating LLM recommendations: {e}. Terminating program.")
            sys.exit(1)
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract recommended items"""
        lines = response.split('\n')
        recommendations = []
        
        for line in lines:
            if 'Movie ID' in line:
                recommendations.append(line.strip())
        
        return recommendations[:10]
    
    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'config': self.config,
            'user_vectors': self.user_vectors,
            'item_vectors': self.item_vectors,
            'transfer_matrix_A': self.transfer_matrix_A,
            'projection_matrix_B': self.projection_matrix_B,
            'context_weights': self.context_weights,
            'item_biases': self.item_biases,
            'online_factors': self.online_learner.online_factors,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.user_vectors = model_data['user_vectors']
        self.item_vectors = model_data['item_vectors']
        self.transfer_matrix_A = model_data['transfer_matrix_A']
        self.projection_matrix_B = model_data['projection_matrix_B']
        self.context_weights = model_data['context_weights']
        self.item_biases = model_data['item_biases']
        self.online_learner.online_factors = model_data['online_factors']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")