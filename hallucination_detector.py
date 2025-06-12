import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import time
import random

from utils import safe_llm_invoke

class HallucinationDetector:
    """Detect and mitigate hallucinations in LLM recommendations with improved rate limiting"""
    
    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # MovieLens genres
        self.all_genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                          'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Cache for genre extractions to avoid repeated LLM calls
        self._genre_extraction_cache = {}
        
        # Fallback feature patterns for genre detection
        self._genre_keywords = {
            'Action': ['action', 'fight', 'battle', 'explosive', 'combat', 'adventure'],
            'Comedy': ['comedy', 'funny', 'humor', 'laugh', 'hilarious', 'amusing'],
            'Drama': ['drama', 'emotional', 'dramatic', 'intense', 'serious'],
            'Horror': ['horror', 'scary', 'frightening', 'terror', 'haunted'],
            'Romance': ['romance', 'love', 'romantic', 'relationship', 'dating'],
            'Sci-Fi': ['sci-fi', 'science fiction', 'future', 'space', 'alien', 'technology'],
            'Thriller': ['thriller', 'suspense', 'tension', 'mystery', 'psychological'],
            'Animation': ['animation', 'animated', 'cartoon', 'disney', 'pixar'],
            'Documentary': ['documentary', 'real', 'true story', 'factual'],
            'Musical': ['musical', 'music', 'songs', 'singing', 'broadway']
        }
    
    def detect_and_mitigate(self, llm_recommendations: List[str],
                          candidate_items: List[int],
                          user_id: int,
                          item_features: np.ndarray,
                          user_vectors: np.ndarray,
                          transfer_matrix_A: np.ndarray,
                          projection_matrix_B: np.ndarray,
                          online_factors: Dict[int, np.ndarray],
                          context_weights: np.ndarray,
                          item_biases: np.ndarray,
                          item_metadata: Dict[int, Dict]) -> List[int]:
        """Detect and mitigate hallucinations with improved efficiency"""
        
        final_recommendations = []
        hallucination_count = 0
        
        for recommendation in llm_recommendations:
            # Check if recommendation exists in candidate set
            item_id = self._extract_item_id(recommendation, candidate_items, item_metadata)
            
            if item_id is not None and item_id in candidate_items:
                # Valid recommendation
                final_recommendations.append(item_id)
            else:
                # Hallucination detected
                hallucination_count += 1
                replacement_id = self._find_replacement(
                    recommendation, candidate_items, user_id, item_features,
                    user_vectors, transfer_matrix_A, projection_matrix_B,
                    online_factors, context_weights, item_biases, item_metadata
                )
                if replacement_id is not None:
                    final_recommendations.append(replacement_id)
        
        if hallucination_count > 0:
            print(f"Detected and mitigated {hallucination_count} hallucinations")
        
        return final_recommendations[:self.config.top_k]
    
    def _extract_item_id(self, recommendation: str, candidate_items: List[int],
                        item_metadata: Dict[int, Dict]) -> Optional[int]:
        """Extract item ID from LLM recommendation"""
        # Try to find movie title in recommendation
        recommendation_lower = recommendation.lower()
        
        for item_id in candidate_items:
            if item_id in item_metadata:
                title = item_metadata[item_id]['title'].lower()
                # Remove year from title for better matching
                title_clean = re.sub(r'\(\d{4}\)', '', title).strip()
                
                if title_clean in recommendation_lower or any(word in recommendation_lower for word in title_clean.split() if len(word) > 3):
                    return item_id
        
        # Try to extract item number if mentioned
        numbers = re.findall(r'\b\d+\b', recommendation)
        for num_str in numbers:
            item_id = int(num_str) - 1  # Convert to 0-based indexing
            if item_id in candidate_items:
                return item_id
        
        return None
    
    def _find_replacement(self, hallucinated_recommendation: str,
                         candidate_items: List[int],
                         user_id: int,
                         item_features: np.ndarray,
                         user_vectors: np.ndarray,
                         transfer_matrix_A: np.ndarray,
                         projection_matrix_B: np.ndarray,
                         online_factors: Dict[int, np.ndarray],
                         context_weights: np.ndarray,
                         item_biases: np.ndarray,
                         item_metadata: Dict[int, Dict]) -> Optional[int]:
        """Find optimal replacement for hallucinated item with improved feature extraction"""
        
        # Extract pseudo-features from hallucinated description
        pseudo_features = self._extract_pseudo_features_improved(hallucinated_recommendation)
        
        best_item = None
        best_score = -np.inf
        
        user_vec = user_vectors[user_id]
        
        for item_id in candidate_items:
            # Calculate genre compatibility
            genre_similarity = self._calculate_genre_similarity(
                pseudo_features, item_features[item_id]
            )
            
            # Calculate user experience factor
            user_experience = self._calculate_user_experience(user_id, user_vectors)
            
            # Adaptive balance parameter
            alpha = genre_similarity * (1 - user_experience)
            
            # Calculate semantic similarity
            semantic_sim = cosine_similarity(
                pseudo_features.reshape(1, -1),
                item_features[item_id].reshape(1, -1)
            )[0, 0]
            
            # Calculate predicted rating
            predicted_rating = self._predict_rating_for_replacement(
                user_vec, item_id, item_features, transfer_matrix_A,
                projection_matrix_B, online_factors, context_weights, item_biases
            )
            
            # Combined score
            score = alpha * semantic_sim + (1 - alpha) * predicted_rating
            
            if score > best_score:
                best_score = score
                best_item = item_id
        
        return best_item
    
    def _extract_pseudo_features_improved(self, description: str) -> np.ndarray:
        """Extract pseudo-features with fallback methods to avoid LLM calls when possible"""
        
        # Check cache first
        cache_key = description.lower().strip()
        if cache_key in self._genre_extraction_cache:
            return self._genre_extraction_cache[cache_key]
        
        # Try keyword-based extraction first
        keyword_genres = self._extract_genres_by_keywords(description)
        
        # If keyword extraction is confident (found multiple genres), use it
        if np.sum(keyword_genres) >= 2:
            print("Using keyword-based genre extraction (avoiding LLM call)")
            pseudo_features = self._create_pseudo_features(keyword_genres)
            self._genre_extraction_cache[cache_key] = pseudo_features
            return pseudo_features
        
        # Fall back to LLM extraction for ambiguous cases
        if self.llm is not None:
            try:
                llm_genres = self._extract_genres_with_llm(description)
                pseudo_features = self._create_pseudo_features(llm_genres)
                self._genre_extraction_cache[cache_key] = pseudo_features
                return pseudo_features
            except Exception as e:
                print(f"LLM genre extraction failed: {e}, using keyword fallback")
        
        # Final fallback: use keyword extraction even if not confident
        pseudo_features = self._create_pseudo_features(keyword_genres)
        self._genre_extraction_cache[cache_key] = pseudo_features
        return pseudo_features
    
    def _extract_genres_by_keywords(self, description: str) -> np.ndarray:
        """Extract genres using keyword matching (no LLM needed)"""
        description_lower = description.lower()
        genre_vector = np.zeros(len(self.all_genres))
        
        for i, genre in enumerate(self.all_genres):
            if genre.lower() in self._genre_keywords:
                keywords = self._genre_keywords[genre.lower()]
                # Check if any keywords match
                if any(keyword in description_lower for keyword in keywords):
                    genre_vector[i] = 1.0
            
            # Direct genre name matching
            if genre.lower().replace("'", "") in description_lower:
                genre_vector[i] = 1.0
        
        return genre_vector
    
    def _extract_genres_with_llm(self, description: str) -> np.ndarray:
        """Extract genres using LLM (with improved rate limiting)"""
        genre_prompt = f"""Given this movie description: {description}
        
Identify which of these genres apply: {', '.join(self.all_genres)}

Return only the applicable genres as a comma-separated list. If unsure, make reasonable guesses based on the description."""
        
        # Use improved safe LLM call with longer delay
        extracted_text = safe_llm_invoke(self.llm, genre_prompt).lower()
        
        # Parse LLM response
        genre_vector = np.zeros(len(self.all_genres))
        for i, genre in enumerate(self.all_genres):
            if genre.lower() in extracted_text or genre.lower().replace("'", "") in extracted_text:
                genre_vector[i] = 1.0
        
        return genre_vector
    
    def _create_pseudo_features(self, genre_vector: np.ndarray) -> np.ndarray:
        """Create complete pseudo-feature vector"""
        # Ensure genre vector has correct length
        if len(genre_vector) != len(self.all_genres):
            print(f"Warning: Genre vector length mismatch, using default")
            genre_vector = np.zeros(len(self.all_genres))
            genre_vector[self.all_genres.index('Drama')] = 1.0  # Default to Drama
        
        # Add normalized year feature (default to mid-range)
        year_feature = np.array([0.5])
        
        # Combine features
        pseudo_features = np.concatenate([genre_vector, year_feature])
        return pseudo_features
    
    def _extract_pseudo_features_safe(self, description: str) -> np.ndarray:
        """Legacy method maintained for compatibility"""
        return self._extract_pseudo_features_improved(description)
    
    def _calculate_genre_similarity(self, pseudo_features: np.ndarray, 
                                  item_features: np.ndarray) -> float:
        """Calculate Jaccard similarity for genres"""
        # Extract genre features (first 19 features)
        pseudo_genres = pseudo_features[:19] > 0
        item_genres = item_features[:19] > 0
        
        intersection = np.sum(pseudo_genres & item_genres)
        union = np.sum(pseudo_genres | item_genres)
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def _calculate_user_experience(self, user_id: int, user_vectors: np.ndarray) -> float:
        """Calculate normalized user experience"""
        # Use user vector norm as proxy for experience
        user_norm = np.linalg.norm(user_vectors[user_id])
        mean_norm = np.mean([np.linalg.norm(u) for u in user_vectors])
        
        return min(1.0, user_norm / mean_norm)
    
    def _predict_rating_for_replacement(self, user_vec: np.ndarray, item_id: int,
                                      item_features: np.ndarray,
                                      transfer_matrix_A: np.ndarray,
                                      projection_matrix_B: np.ndarray,
                                      online_factors: Dict[int, np.ndarray],
                                      context_weights: np.ndarray,
                                      item_biases: np.ndarray) -> float:
        """Predict rating for replacement candidate"""
        # Content-based score
        content_score = user_vec @ transfer_matrix_A @ item_features[item_id]
        
        # Online adaptation score
        online_score = 0.0
        if item_id in online_factors:
            online_score = user_vec @ projection_matrix_B @ online_factors[item_id]
        
        # Bias
        bias = item_biases[item_id]
        
        # Combined prediction (normalized to [0,1])
        linear_score = content_score + online_score + bias
        predicted_rating = 1 + 4 * self._sigmoid(linear_score)
        return predicted_rating / 5.0  # Normalize to [0,1]
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the genre extraction cache"""
        return {
            'cache_size': len(self._genre_extraction_cache),
            'cache_hits': getattr(self, '_cache_hits', 0),
            'llm_calls': getattr(self, '_llm_calls', 0)
        }
    
    def clear_cache(self):
        """Clear the genre extraction cache"""
        self._genre_extraction_cache.clear()
        print("Genre extraction cache cleared")
    
    def preload_cache_from_training_data(self, item_metadata: Dict[int, Dict]):
        """Preload cache with training data to avoid LLM calls during testing"""
        print("Preloading genre extraction cache from training data...")
        
        for item_id, metadata in item_metadata.items():
            title = metadata.get('title', '')
            genres = metadata.get('genres', [])
            
            # Create a description from title and known genres
            description = f"{title} - {', '.join(genres)}"
            
            # Create genre vector from known genres
            genre_vector = np.zeros(len(self.all_genres))
            for genre in genres:
                if genre in self.all_genres:
                    genre_vector[self.all_genres.index(genre)] = 1.0
            
            # Cache this mapping
            cache_key = description.lower().strip()
            pseudo_features = self._create_pseudo_features(genre_vector)
            self._genre_extraction_cache[cache_key] = pseudo_features
        
        print(f"Preloaded {len(self._genre_extraction_cache)} entries in genre cache")