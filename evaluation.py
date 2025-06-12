import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class RecommendationEvaluator:
    """Evaluation metrics for recommendation systems"""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values
    
    def evaluate_recommendations(self, test_interactions: np.ndarray,
                               predictions: np.ndarray,
                               user_recommendations: Dict[int, List[int]],
                               test_df: pd.DataFrame) -> Dict[str, float]:
        """Comprehensive evaluation of recommendations"""
        results = {}
        
        # Rating prediction metrics
        results.update(self._evaluate_rating_prediction(test_interactions, predictions))
        
        # Ranking metrics
        results.update(self._evaluate_ranking(user_recommendations, test_df))
        
        # Coverage and diversity
        results.update(self._evaluate_coverage_diversity(user_recommendations, test_interactions))
        
        return results
    
    def _evaluate_rating_prediction(self, test_interactions: np.ndarray,
                                  predictions: np.ndarray) -> Dict[str, float]:
        """Evaluate rating prediction accuracy"""
        # Get observed test ratings
        observed_mask = test_interactions > 0
        true_ratings = test_interactions[observed_mask]
        pred_ratings = predictions[observed_mask]
        
        # Clip predictions to valid range
        pred_ratings = np.clip(pred_ratings, 1, 5)
        
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae = mean_absolute_error(true_ratings, pred_ratings)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def _evaluate_ranking(self, user_recommendations: Dict[int, List[int]],
                         test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ranking quality"""
        results = {}
        
        # Create test set for each user
        user_test_items = {}
        for _, row in test_df.iterrows():
            user_id = row['user_id'] - 1  # Convert to 0-based
            item_id = row['item_id'] - 1
            rating = row['rating']
            
            if user_id not in user_test_items:
                user_test_items[user_id] = []
            
            # Only consider items with rating >= 4 as relevant
            if rating >= 4:
                user_test_items[user_id].append(item_id)
        
        # Calculate metrics for each k
        for k in self.k_values:
            hit_rates = []
            ndcgs = []
            mrrs = []
            
            for user_id, test_items in user_test_items.items():
                if user_id in user_recommendations and len(test_items) > 0:
                    recs = user_recommendations[user_id][:k]
                    
                    # Hit Rate@K
                    hits = len(set(recs) & set(test_items))
                    hit_rate = hits / min(len(test_items), k)
                    hit_rates.append(hit_rate)
                    
                    # NDCG@K
                    ndcg = self._calculate_ndcg(recs, test_items, k)
                    ndcgs.append(ndcg)
                    
                    # MRR
                    mrr = self._calculate_mrr(recs, test_items)
                    mrrs.append(mrr)
            
            if hit_rates:
                results[f'HitRate@{k}'] = np.mean(hit_rates)
                results[f'NDCG@{k}'] = np.mean(ndcgs)
                results[f'MRR@{k}'] = np.mean(mrrs)
            else:
                results[f'HitRate@{k}'] = 0.0
                results[f'NDCG@{k}'] = 0.0
                results[f'MRR@{k}'] = 0.0
        
        return results
    
    def _calculate_ndcg(self, recommendations: List[int], 
                       relevant_items: List[int], k: int) -> float:
        """Calculate NDCG@K"""
        # Create relevance scores
        relevance = [1 if item in relevant_items else 0 for item in recommendations[:k]]
        
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        
        # IDCG
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, recommendations: List[int], 
                      relevant_items: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, item in enumerate(recommendations):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def _evaluate_coverage_diversity(self, user_recommendations: Dict[int, List[int]],
                                   test_interactions: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage and diversity"""
        all_recommendations = []
        for recs in user_recommendations.values():
            all_recommendations.extend(recs)
        
        # Catalog coverage
        unique_items = len(set(all_recommendations))
        total_items = test_interactions.shape[1]
        catalog_coverage = unique_items / total_items
        
        # Average diversity (simplified as unique items per user)
        diversities = [len(set(recs)) / len(recs) if recs else 0 
                      for recs in user_recommendations.values()]
        avg_diversity = np.mean(diversities) if diversities else 0
        
        return {
            'CatalogCoverage': catalog_coverage,
            'AvgDiversity': avg_diversity
        }
    
    def evaluate_hallucination_rate(self, llm_recommendations: List[List[str]],
                                  valid_items: List[int]) -> float:
        """Calculate hallucination rate"""
        total_recommendations = 0
        hallucinated_recommendations = 0
        
        for user_recs in llm_recommendations:
            for rec in user_recs:
                total_recommendations += 1
                # Check if recommendation corresponds to a valid item
                # This is simplified - you'd implement proper matching logic
                if not any(str(item_id) in rec for item_id in valid_items):
                    hallucinated_recommendations += 1
        
        return hallucinated_recommendations / total_recommendations if total_recommendations > 0 else 0.0
    
    def print_results(self, results: Dict[str, float]):
        """Print evaluation results"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Rating prediction
        print("\nRating Prediction:")
        print(f"  RMSE: {results.get('RMSE', 0):.4f}")
        print(f"  MAE:  {results.get('MAE', 0):.4f}")
        
        # Ranking metrics
        print("\nRanking Metrics:")
        for k in [5, 10, 20]:
            if f'HitRate@{k}' in results:
                print(f"  Hit Rate@{k}:  {results[f'HitRate@{k}']:.4f}")
                print(f"  NDCG@{k}:      {results[f'NDCG@{k}']:.4f}")
                print(f"  MRR@{k}:       {results[f'MRR@{k}']:.4f}")
        
        # Coverage and diversity
        print("\nCoverage & Diversity:")
        print(f"  Catalog Coverage: {results.get('CatalogCoverage', 0):.4f}")
        print(f"  Avg Diversity:    {results.get('AvgDiversity', 0):.4f}")
        
        print("\n" + "=" * 60)