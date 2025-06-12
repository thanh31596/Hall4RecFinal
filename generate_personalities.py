#!/usr/bin/env python3
"""
Personality Profile Pre-Generation Script
Pre-generates personality profiles for all users using LLM with robust rate limiting
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Import your existing modules
from data_loader import MovieLensDataLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from google.api_core.exceptions import ResourceExhausted
# Gemini 2.0 Flash rate limits (from Google documentation)
GEMINI_RATE_LIMITS = {
    "gemini-2.0-flash-001": {
        "rpm": 15,      # Requests per minute
        "tpm": 1000000, # Tokens per minute
        "rpd": 1500,    # Requests per day
        "safe_delay": 4.5,  # Seconds between requests (60/15 = 4.0, +0.5 safety margin)
        "batch_delay": 20.0 # Longer delay between batches
    },
    "gemini-2.5-flash": {
        "rpm": 10,
        "tpm": 250000,
        "rpd": 500,
        "safe_delay": 6.5,   # 60/10 = 6.0, +0.5 safety margin
        "batch_delay": 25.0
    },
    "gemini-2.5-pro": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 25,
        "safe_delay": 12.5,  # 60/5 = 12.0, +0.5 safety margin
        "batch_delay": 30.0
    }
}

class PersonalityProfileGenerator:
    """Pre-generates personality profiles with robust rate limiting"""
    
    def __init__(self, output_path: str = "./personalities.json", resume: bool = True):
        self.output_path = Path(output_path)
        self.resume = resume
        self.sentence_bert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM with conservative settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0,
            #max_tokens=500,  # Reduced for faster responses
            timeout=90,      # Increased timeout
            max_retries=1,   # We handle retries manually
            request_timeout=60,
        )
        
        # Rate limiting parameters
        self.base_delay = 3.0  # Base delay between requests
        self.batch_delay = 15.0  # Delay between batches
        self.max_delay = 120.0  # Maximum delay for backoff
        
        # Load existing data if resuming
        self.existing_data = self._load_existing_data() if resume else {}
        
        print(f"Personality Generator initialized")
        print(f"Output path: {self.output_path}")
        print(f"Resume mode: {resume}")
        if resume and self.existing_data:
            completed = len([p for p in self.existing_data.get('personalities', {}).values() 
                           if p.get('generation_status') == 'success'])
            print(f"Found {completed} existing personalities")
    
    def _load_existing_data(self) -> Dict:
        """Load existing personality data if available"""
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded existing data with {len(data.get('personalities', {}))} entries")
                return data
            except Exception as e:
                print(f"Error loading existing data: {e}")
        return {}
    
    def _save_data(self, data: Dict):
        """Save data with backup"""
        # Create backup if file exists
        if self.output_path.exists():
            backup_path = self.output_path.with_suffix(f'.backup_{int(time.time())}.json')
            try:
                import shutil
                shutil.copy2(self.output_path, backup_path)
                print(f"Created backup: {backup_path}")
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
        
        # Save new data
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Data saved to {self.output_path}")
    
    @retry(
        retry=retry_if_exception_type((ResourceExhausted, Exception)),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        stop=stop_after_attempt(5)
    )
    def _safe_llm_call(self, prompt: str, delay_before: float = 3.0) -> str:
        """Make a safe LLM call with rate limiting"""
        # Wait before making the call
        print(f"Waiting {delay_before:.1f}s before LLM call...")
        time.sleep(delay_before)
        
        try:
            print("Prompt being sent:")
            print("="*50)
            print(prompt)
            print("="*50)
            print(f"Prompt length: {len(prompt)} characters")
            print("Making LLM call...")
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            print("LLM Response: ",response)
            print("result LLM: ", result)
            # Additional delay after successful call
            time.sleep(random.uniform(1.0, 2.0))
            print("LLM call successful")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                print(f"Rate limit hit: {e}")
                # Exponential backoff for rate limits
                delay = min(self.max_delay, self.base_delay * (2 ** random.uniform(1, 3)))
                print(f"Backing off for {delay:.1f} seconds...")
                time.sleep(delay)
            raise e
    
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
    
    def _validate_llm_response(self, response: str) -> bool:
        """Validate LLM response quality"""
        if not response or len(response.strip()) < 20:
            return False
        if "unable to generate" in response.lower() or "api limitations" in response.lower():
            return False
        return True
    
    def generate_personality_profile(self, user_id: int, demographics: Dict, 
                                   genre_prefs: Dict[str, float]) -> Dict[str, Any]:
        """Generate personality profile for a single user"""
        print(f"\n--- Generating personality for User {user_id} ---")
        
        # Create base personality description
        personality_desc = self._create_personality_description(demographics, genre_prefs)
        print(f"Base description: {personality_desc[:100]}...")
        
        # Create reflection prompt
        reflection_prompt = f"""Based on this user profile: {personality_desc}

Generate insights about this user's movie preferences that go beyond simple genre statistics. 
Consider their personality traits, lifestyle, and what drives their entertainment choices.
Focus on deeper patterns and decision-making factors for movie selection.
Keep response under 100 words and be specific."""
        
        try:
            # Generate reflection with progressive delay increase
            base_delay = self.base_delay + random.uniform(0, 2)
            reflection = self._safe_llm_call(reflection_prompt, delay_before=base_delay)
            
            # Validate response
            if not self._validate_llm_response(reflection):
                print("TRUE REFLECTION: ",reflection)
                print("Warning: LLM response failed validation, using fallback")
                reflection = f"User enjoys diverse entertainment options and values engaging storytelling based on their {demographics.get('occupation', 'lifestyle')} background."
            
            # Synthesize final summary
            summary = f"{personality_desc}\n\nPersonality Insights: {reflection}"
            
            # Generate embedding
            print("Generating embedding...")
            embedding = self.sentence_bert.encode(summary).tolist()
            
            result = {
                'user_id': user_id,
                'demographics': demographics,
                'genre_preferences': genre_prefs,
                'personality_description': personality_desc,
                'reflection': reflection,
                'summary': summary,
                'embedding': embedding,
                'generation_status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ User {user_id} personality generated successfully")
            return result
            
        except Exception as e:
            print(f"✗ Error generating personality for User {user_id}: {e}")
            
            # Return fallback personality
            fallback_reflection = f"User enjoys diverse entertainment options and values engaging storytelling based on their {demographics.get('occupation', 'lifestyle')} background."
            fallback_summary = f"{personality_desc}\n\nPersonality Insights: {fallback_reflection}"
            fallback_embedding = self.sentence_bert.encode(fallback_summary).tolist()
            
            return {
                'user_id': user_id,
                'demographics': demographics,
                'genre_preferences': genre_prefs,
                'personality_description': personality_desc,
                'reflection': fallback_reflection,
                'summary': fallback_summary,
                'embedding': fallback_embedding,
                'generation_status': 'fallback',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_all_personalities(self, data_path: str = "./ml-100k/", 
                                 batch_size: int = 5) -> Dict[str, Any]:
        """Generate personalities for all users with robust rate limiting"""
        print("="*60)
        print("STARTING PERSONALITY PROFILE GENERATION")
        print("="*60)
        
        # Load data
        print("Loading MovieLens data...")
        data_loader = MovieLensDataLoader(data_path)
        data = data_loader.load_all_data()
        
        train_df, _ = data_loader.train_test_split(test_ratio=0.2)
        train_matrix = data_loader.create_interaction_matrix(train_df)
        user_demographics = data_loader.create_user_demographics()
        item_metadata = data_loader.create_item_metadata()
        
        n_users = train_matrix.shape[0]
        print(f"Loaded data for {n_users} users")
        
        # Initialize result structure
        result_data = self.existing_data if self.existing_data else {
            'metadata': {
                'dataset': 'movielens-100k',
                'generation_timestamp': datetime.now().isoformat(),
                'total_users': n_users,
                'llm_model': 'gemini-2.0-flash-001',
                'batch_size': batch_size,
                'base_delay': self.base_delay,
                'batch_delay': self.batch_delay
            },
            'personalities': {}
        }
        
        # Track progress
        completed_users = set()
        if 'personalities' in result_data:
            completed_users = {int(uid) for uid, profile in result_data['personalities'].items() 
                             if profile.get('generation_status') in ['success', 'fallback']}
        
        remaining_users = [uid for uid in range(n_users) if uid not in completed_users]
        
        print(f"Progress: {len(completed_users)}/{n_users} users completed")
        print(f"Remaining: {len(remaining_users)} users")
        
        if not remaining_users:
            print("All users already completed!")
            return result_data
        
        # Process remaining users in batches
        start_time = time.time()
        
        for batch_start in range(0, len(remaining_users), batch_size):
            batch_users = remaining_users[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(remaining_users) + batch_size - 1) // batch_size
            
            print(f"\n{'='*50}")
            print(f"BATCH {batch_num}/{total_batches}")
            print(f"Users: {batch_users}")
            print(f"{'='*50}")
            
            # Process each user in the batch
            for i, user_id in enumerate(batch_users):
                try:
                    demographics = user_demographics.get(user_id, {})
                    genre_prefs = self._extract_genre_preferences(user_id, train_matrix, item_metadata)
                    
                    # Generate personality
                    personality = self.generate_personality_profile(user_id, demographics, genre_prefs)
                    
                    # Store result
                    result_data['personalities'][str(user_id)] = personality
                    
                    # Save after each successful generation
                    self._save_data(result_data)
                    
                    # Progress update
                    completed = len([p for p in result_data['personalities'].values() 
                                   if p.get('generation_status') in ['success', 'fallback']])
                    elapsed = time.time() - start_time
                    avg_time_per_user = elapsed / max(1, completed - len(completed_users))
                    remaining = n_users - completed
                    eta = avg_time_per_user * remaining
                    
                    print(f"Progress: {completed}/{n_users} users ({completed/n_users*100:.1f}%)")
                    print(f"ETA: {eta/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"Critical error processing user {user_id}: {e}")
                    # Continue with next user
                    continue
            
            # Batch-level delay (except for last batch)
            if batch_start + batch_size < len(remaining_users):
                delay = self.batch_delay + random.uniform(0, 5)
                print(f"\n⏳ Batch completed. Waiting {delay:.1f}s before next batch...")
                time.sleep(delay)
        
        # Final save and summary
        result_data['metadata']['completion_timestamp'] = datetime.now().isoformat()
        result_data['metadata']['total_time_seconds'] = time.time() - start_time
        self._save_data(result_data)
        
        # Summary
        successful = len([p for p in result_data['personalities'].values() 
                         if p.get('generation_status') == 'success'])
        fallback = len([p for p in result_data['personalities'].values() 
                       if p.get('generation_status') == 'fallback'])
        
        print("\n" + "="*60)
        print("PERSONALITY GENERATION COMPLETED!")
        print("="*60)
        print(f"Total users: {n_users}")
        print(f"Successful generations: {successful}")
        print(f"Fallback generations: {fallback}")
        print(f"Success rate: {successful/n_users*100:.1f}%")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Output saved to: {self.output_path}")
        print("="*60)
        
        return result_data

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Pre-generate personality profiles for HallAgent4Rec')
    parser.add_argument('--data_path', default='./ml-100k/', help='Path to MovieLens 100K data')
    parser.add_argument('--output', default='./personalities.json', help='Output JSON file path')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--no_resume', action='store_true', help='Start fresh (don\'t resume)')
    parser.add_argument('--base_delay', type=float, default=3.0, help='Base delay between requests')
    parser.add_argument('--batch_delay', type=float, default=15.0, help='Delay between batches')
    
    args = parser.parse_args()
    
    # Verify API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("Error: GOOGLE_API_KEY environment variable not set!")
        print("Please set it with: export GOOGLE_API_KEY='your_api_key'")
        sys.exit(1)
    
    # Initialize generator
    generator = PersonalityProfileGenerator(
        output_path=args.output,
        resume=not args.no_resume
    )
    
    # Update delays if specified
    generator.base_delay = args.base_delay
    generator.batch_delay = args.batch_delay
    
    try:
        # Generate personalities
        result = generator.generate_all_personalities(
            data_path=args.data_path,
            batch_size=args.batch_size
        )
        
        print("\n🎉 Generation completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        print("Progress has been saved. You can resume by running the script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
