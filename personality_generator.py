import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import random
import json
from pathlib import Path

from utils import safe_llm_invoke, batch_llm_calls

class PersonalityVectorGenerator:
    """Load personality vectors from pre-generated JSON or generate on-demand"""
    
    def __init__(self, llm=None, embeddings_model=None):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.sentence_bert = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_personality_vectors_from_json(self, json_path: str = "./personalities.json") -> np.ndarray:
        """Load personality vectors from pre-generated JSON file"""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Personality JSON file not found: {json_path}")
        
        print(f"Loading personality vectors from {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            personalities = data.get('personalities', {})
            if not personalities:
                raise ValueError("No personalities found in JSON file")
            
            # Extract embeddings in user ID order
            n_users = len(personalities)
            embedding_dim = len(list(personalities.values())[0]['embedding'])
            
            personality_vectors = np.zeros((n_users, embedding_dim))
            
            successful_loads = 0
            fallback_loads = 0
            
            for user_id in range(n_users):
                user_key = str(user_id)
                if user_key in personalities:
                    personality_data = personalities[user_key]
                    
                    # Check generation status
                    status = personality_data.get('generation_status', 'unknown')
                    if status == 'success':
                        successful_loads += 1
                    elif status == 'fallback':
                        fallback_loads += 1
                    
                    # Load embedding
                    embedding = personality_data.get('embedding', [])
                    if len(embedding) == embedding_dim:
                        personality_vectors[user_id] = np.array(embedding)
                    else:
                        print(f"Warning: Invalid embedding for user {user_id}, using random")
                        personality_vectors[user_id] = np.random.normal(0, 0.1, embedding_dim)
                else:
                    print(f"Warning: No personality data for user {user_id}, using random")
                    personality_vectors[user_id] = np.random.normal(0, 0.1, embedding_dim)
            
            # Log statistics
            metadata = data.get('metadata', {})
            generation_time = metadata.get('generation_timestamp', 'unknown')
            
            print(f"✓ Loaded personality vectors for {n_users} users")
            print(f"  - Successful generations: {successful_loads}")
            print(f"  - Fallback generations: {fallback_loads}")
            print(f"  - Missing/Random: {n_users - successful_loads - fallback_loads}")
            print(f"  - Generated on: {generation_time}")
            print(f"  - Embedding dimension: {embedding_dim}")
            
            return personality_vectors
            
        except Exception as e:
            raise RuntimeError(f"Error loading personality vectors from JSON: {e}")
    
    def generate_personality_vectors(self, user_demographics: Dict[int, Dict],
                                   interaction_matrix: np.ndarray,
                                   item_metadata: Dict[int, Dict],
                                   json_path: Optional[str] = "./personalities.json") -> np.ndarray:
        """
        Load personality vectors from JSON if available, otherwise generate on-demand
        
        Args:
            user_demographics: User demographic information
            interaction_matrix: User-item interaction matrix
            item_metadata: Item metadata information
            json_path: Path to pre-generated personality JSON file
            
        Returns:
            np.ndarray: Personality vectors for all users
        """
        
        # Try to load from JSON first
        if json_path:
            json_path_obj = Path(json_path)
            if json_path_obj.exists():
                try:
                    print("Attempting to load pre-generated personality vectors...")
                    return self.load_personality_vectors_from_json(json_path)
                except Exception as e:
                    print(f"Warning: Failed to load from JSON ({e})")
                    print("Falling back to on-demand generation...")
            else:
                print(f"Personality JSON not found at {json_path}")
                print("Falling back to on-demand generation...")
        
        # Fallback to on-demand generation
        return self._generate_personality_vectors_on_demand(
            user_demographics, interaction_matrix, item_metadata
        )
    
    def _generate_personality_vectors_on_demand(self, user_demographics: Dict[int, Dict],
                                              interaction_matrix: np.ndarray,
                                              item_metadata: Dict[int, Dict]) -> np.ndarray:
        """Generate personality vectors on-demand (original implementation with improved rate limiting)"""
        if not self.llm:
            raise ValueError("LLM not provided for on-demand generation. Please use pre-generated personalities.")
        
        n_users = interaction_matrix.shape[0]
        personality_vectors = []
        
        print("Generating personality vectors on-demand...")
        print("⚠️  Warning: This may take a long time and could hit rate limits!")
        print("   Consider using the pre-generation script: python generate_personalities.py")
        
        # Process users in smaller batches with longer delays
        batch_size = 3  # Smaller batches
        for batch_start in range(0, n_users, batch_size):
            batch_end = min(batch_start + batch_size, n_users)
            batch_users = range(batch_start, batch_end)
            
            print(f"Processing users {batch_start}-{batch_end-1}/{n_users}")
            
            # Prepare batch of prompts
            batch_prompts = []
            batch_descriptions = []
            
            for user_id in batch_users:
                # Extract genre preferences from interaction history
                genre_prefs = self._extract_genre_preferences(user_id, interaction_matrix, item_metadata)
                
                # Get user demographics
                demographics = user_demographics.get(user_id, {})
                
                # Create personality description
                personality_desc = self._create_personality_description(demographics, genre_prefs)
                batch_descriptions.append(personality_desc)
                
                # Create reflection prompt
                reflection_prompt = f"""Based on this user profile: {personality_desc}

Generate insights about this user's movie preferences that go beyond simple genre statistics. 
Consider their personality traits, lifestyle, and what drives their entertainment choices.
Focus on deeper patterns and decision-making factors for movie selection.
Keep response under 100 words."""
                
                batch_prompts.append(reflection_prompt)
            
            # Generate reflections in batch with extended delays
            try:
                batch_reflections = batch_llm_calls(
                    self.llm, 
                    batch_prompts, 
                    batch_size=1,  # Process one at a time
                    delay_between_batches=20.0  # Longer delays
                )
            except Exception as e:
                print(f"Error in batch LLM generation: {e}")
                # Fallback to default reflections
                batch_reflections = [
                    "User enjoys diverse entertainment options and values engaging storytelling." 
                    for _ in batch_prompts
                ]
            
            # Process batch results
            for i, user_id in enumerate(batch_users):
                personality_desc = batch_descriptions[i]
                reflection = batch_reflections[i] if i < len(batch_reflections) else "User enjoys diverse entertainment options."
                
                # Synthesize final summary
                summary = f"{personality_desc}\n\nPersonality Insights: {reflection}"
                
                # Convert to embedding using SentenceBERT
                try:
                    embedding = self.sentence_bert.encode(summary)
                    personality_vectors.append(embedding)
                except Exception as e:
                    print(f"Error encoding summary for user {user_id}: {e}")
                    # Fallback to random embedding
                    embedding = np.random.normal(0, 0.1, 384)  # Default SentenceBERT dimension
                    personality_vectors.append(embedding)
            
            # Progress update and longer delay between batches
            if batch_end < n_users:
                print(f"Completed {batch_end}/{n_users} users. Waiting 30s before next batch...")
                time.sleep(30)  # Much longer delay
        
        print("On-demand personality vector generation completed!")
        return np.array(personality_vectors)
    
    def _extract_genre_preferences(self, user_id: int, interaction_matrix: np.ndarray,
                                 item_metadata: Dict[int, Dict]) -> Dict[str, float]:
        """Extract genre preferences from user's interaction history"""
        user_interactions = interaction_matrix[user_id]
        genre_counts = {}
        total_weighted_interactions = 0
        
        for item_id, rating in enumerate(user_interactions):
            if rating > 0:  # User interacted with this item
                item_meta = item_metadata.get(item_id, {})
                genres = item_meta.get('genres', [])
                
                # Weight by rating (higher ratings indicate stronger preference)
                weight = rating / 5.0  # Normalize to [0,1]
                
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + weight
                total_weighted_interactions += weight
        
        # Normalize to preferences
        if total_weighted_interactions > 0:
            genre_prefs = {genre: count/total_weighted_interactions 
                          for genre, count in genre_counts.items()}
        else:
            genre_prefs = {}
        
        return genre_prefs
    
    def _create_personality_description(self, demographics: Dict, genre_prefs: Dict[str, float]) -> str:
        """Create personality description from demographics and genre preferences"""
        age = demographics.get('age', 'unknown')
        gender = demographics.get('gender', 'unknown')
        occupation = demographics.get('occupation', 'unknown')
        location = demographics.get('zip_code', 'unknown')
        
        # Get top genres
        if genre_prefs:
            top_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
            genre_text = ", ".join([f"{genre} (preference: {pref:.2f})" for genre, pref in top_genres])
        else:
            genre_text = "no clear preferences identified"
        
        description = f"""User is a {age}-year-old {gender} working as {occupation} from zip code {location}. 
        Based on their movie viewing patterns, they show preferences for: {genre_text}."""
        
        return description
    
    def validate_personality_json(self, json_path: str = "./personalities.json") -> Dict[str, Any]:
        """Validate personality JSON file and return statistics"""
        json_path = Path(json_path)
        
        if not json_path.exists():
            return {'status': 'missing', 'path': str(json_path)}
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            personalities = data.get('personalities', {})
            metadata = data.get('metadata', {})
            
            stats = {
                'status': 'valid',
                'path': str(json_path),
                'total_users': len(personalities),
                'successful_generations': len([p for p in personalities.values() 
                                             if p.get('generation_status') == 'success']),
                'fallback_generations': len([p for p in personalities.values() 
                                           if p.get('generation_status') == 'fallback']),
                'generation_timestamp': metadata.get('generation_timestamp'),
                'embedding_dimension': None
            }
            
            # Check embedding dimension
            if personalities:
                first_embedding = list(personalities.values())[0].get('embedding', [])
                stats['embedding_dimension'] = len(first_embedding)
            
            stats['success_rate'] = (stats['successful_generations'] / 
                                   stats['total_users'] * 100) if stats['total_users'] > 0 else 0
            
            return stats
            
        except Exception as e:
            return {
                'status': 'invalid',
                'path': str(json_path),
                'error': str(e)
            }
    
    def print_personality_stats(self, json_path: str = "./personalities.json"):
        """Print statistics about personality JSON file"""
        stats = self.validate_personality_json(json_path)
        
        print("\n" + "="*50)
        print("PERSONALITY DATA STATISTICS")
        print("="*50)
        
        if stats['status'] == 'missing':
            print(f"❌ File not found: {stats['path']}")
            print("   Run: python generate_personalities.py")
        elif stats['status'] == 'invalid':
            print(f"❌ Invalid file: {stats['path']}")
            print(f"   Error: {stats['error']}")
        else:
            print(f"✓ Valid personality data: {stats['path']}")
            print(f"  Total users: {stats['total_users']}")
            print(f"  Successful: {stats['successful_generations']} ({stats['success_rate']:.1f}%)")
            print(f"  Fallback: {stats['fallback_generations']}")
            print(f"  Embedding dim: {stats['embedding_dimension']}")
            print(f"  Generated: {stats.get('generation_timestamp', 'unknown')}")
        
        print("="*50)
        return stats