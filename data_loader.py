import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

class MovieLensDataLoader:
    """Data loader for MovieLens 100K dataset"""
    
    def __init__(self, data_path: str = "./ml-100k/"):
        self.data_path = data_path
        self.ratings_df = None
        self.users_df = None
        self.items_df = None
        self.genres = None
        self.occupations = None
        
    def load_all_data(self) -> Dict[str, Any]:
        """Load all MovieLens 100K data"""
        print("Loading MovieLens 100K dataset...")
        
        # Load ratings
        self.ratings_df = self._load_ratings()
        
        # Load user demographics
        self.users_df = self._load_users()
        
        # Load item information
        self.items_df = self._load_items()
        
        # Load genres and occupations
        self.genres = self._load_genres()
        self.occupations = self._load_occupations()
        
        print(f"Loaded {len(self.ratings_df)} ratings from {len(self.users_df)} users on {len(self.items_df)} items")
        
        return {
            'ratings': self.ratings_df,
            'users': self.users_df,
            'items': self.items_df,
            'genres': self.genres,
            'occupations': self.occupations
        }
    
    def _load_ratings(self) -> pd.DataFrame:
        """Load u.data file"""
        ratings_path = os.path.join(self.data_path, "u.data")
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        
        df = pd.read_csv(ratings_path, sep='\t', names=columns)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def _load_users(self) -> pd.DataFrame:
        """Load u.user file"""
        users_path = os.path.join(self.data_path, "u.user")
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        
        df = pd.read_csv(users_path, sep='|', names=columns)
        return df
    
    def _load_items(self) -> pd.DataFrame:
        """Load u.item file"""
        items_path = os.path.join(self.data_path, "u.item")
        
        # Define column names
        columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url']
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                        'Sci-Fi', 'Thriller', 'War', 'Western']
        columns.extend(genre_columns)
        
        df = pd.read_csv(items_path, sep='|', names=columns, encoding='latin1')
        
        # Parse release date
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        
        return df
    
    def _load_genres(self) -> List[str]:
        """Load u.genre file"""
        genres_path = os.path.join(self.data_path, "u.genre")
        
        with open(genres_path, 'r') as f:
            genres = [line.strip().split('|')[0] for line in f if line.strip()]
        
        return genres
    
    def _load_occupations(self) -> List[str]:
        """Load u.occupation file"""
        occupations_path = os.path.join(self.data_path, "u.occupation")
        
        with open(occupations_path, 'r') as f:
            occupations = [line.strip() for line in f if line.strip()]
        
        return occupations
    
    def create_interaction_matrix(self, ratings_df: pd.DataFrame = None) -> np.ndarray:
        """Create user-item interaction matrix"""
        if ratings_df is None:
            ratings_df = self.ratings_df
        
        # Create pivot table
        interaction_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        ).values
        
        return interaction_matrix
    
    def create_item_features(self) -> np.ndarray:
        """Create item feature matrix"""
        # Genre features (binary)
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                        'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genre_features = self.items_df[genre_columns].values
        
        # Release year feature (normalized)
        release_years = self.items_df['release_year'].fillna(1995).values  # Fill NaN with median year
        year_min, year_max = release_years.min(), release_years.max()
        normalized_years = (release_years - year_min) / (year_max - year_min)
        
        # Combine features
        features = np.hstack([
            genre_features,
            normalized_years.reshape(-1, 1)
        ])
        
        return features
    
    def create_user_demographics(self) -> Dict[int, Dict]:
        """Create user demographics dictionary"""
        demographics = {}
        
        for _, row in self.users_df.iterrows():
            user_id = row['user_id'] - 1  # Convert to 0-based indexing
            demographics[user_id] = {
                'age': row['age'],
                'gender': row['gender'],
                'occupation': row['occupation'],
                'zip_code': row['zip_code']
            }
        
        return demographics
    
    def create_item_metadata(self) -> Dict[int, Dict]:
        """Create item metadata dictionary"""
        metadata = {}
        
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                        'Sci-Fi', 'Thriller', 'War', 'Western']
        
        for _, row in self.items_df.iterrows():
            item_id = row['movie_id'] - 1  # Convert to 0-based indexing
            
            # Get genres for this movie
            genres = [genre for genre in genre_columns if row[genre] == 1]
            
            metadata[item_id] = {
                'title': row['movie_title'],
                'release_year': row['release_year'] if not pd.isna(row['release_year']) else 1995,
                'genres': genres,
                'imdb_url': row['imdb_url']
            }
        
        return metadata
    
    def train_test_split(self, test_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split ratings into train and test sets"""
        train_df, test_df = train_test_split(
            self.ratings_df, 
            test_size=test_ratio, 
            random_state=random_state,
            stratify=self.ratings_df['user_id']  # Ensure each user appears in both sets
        )
        
        return train_df, test_df