import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict
import warnings

# Adapter for HallAgent4Rec to work with RecommendationEvaluator
class HallAgent4RecAdapter:
    """
    Adapter to make HallAgent4Rec compatible with RecommendationEvaluator's expected interface.
    Exposes a recommend(user_id, n_items) method.
    """
    def __init__(self, hallagent, item_features, item_metadata, context_features=None, user_profiles=None):
        self.hallagent = hallagent
        self.item_features = item_features
        self.item_metadata = item_metadata
        self.context_features = context_features
        self.user_profiles = user_profiles or {}

    def recommend(self, user_id, n_items=10):
        # Candidate items: all items except those already interacted with (if available)
        num_items = self.item_features.shape[0]
        candidate_items = list(range(num_items))
        user_profile = self.user_profiles.get(user_id, "")
        # Call HallAgent4Rec's online_recommendation
        recs = self.hallagent.online_recommendation(
            user_id=user_id,
            candidate_items=candidate_items,
            item_features=self.item_features,
            item_metadata=self.item_metadata,
            context_features=self.context_features,
            user_profile=user_profile
        )
        # Return top n_items as item IDs (int)
        # If recs is a list of "Movie ID X: ..." strings, extract the ID
        item_ids = []
        for r in recs:
            if isinstance(r, int):
                item_ids.append(r)
            elif isinstance(r, str) and r.startswith("Movie ID"):
                try:
                    # Extract the number after 'Movie ID ' and subtract 1 (since +1 was used in prompt)
                    item_id = int(r.split()[2].replace(":", "")) - 1
                    item_ids.append(item_id)
                except Exception:
                    continue
        return item_ids[:n_items]

# Example usage:
# evaluator = RecommendationEvaluator()
# adapter = HallAgent4RecAdapter(hallagent, item_features, item_metadata)
# results = evaluator.evaluate_recommendations(recommendations, ground_truth)

class RecommendationEvaluator:
    """
    A comprehensive evaluator for recommendation systems supporting various metrics.
    
    Supports evaluation of:
    - Ranking metrics: Precision@K, Recall@K, F1@K, MAP, MRR, NDCG@K
    - Beyond-accuracy metrics: Coverage, Diversity, Novelty
    - Error metrics: RMSE, MAE (for rating prediction)
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize the evaluator.
        
        Args:
            k_values: List of K values for @K metrics
        """
        self.k_values = k_values
        self.metrics_history = defaultdict(list)
    
    def evaluate_recommendations(self, 
                         recommendations: Dict[int, List[int]], 
                         ground_truth: Dict[int, List[int]],
                         catalog_size: int = None) -> Dict[str, float]:
        """
        Evaluate recommendation rankings against ground truth.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended item_ids (ranked)
            ground_truth: Dict mapping user_id to list of relevant item_ids
            catalog_size: Total number of items in catalog (for coverage calculation)
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        # Calculate metrics for each k value
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ap_scores = []
            rr_scores = []
            ndcg_scores = []
            
            for user_id in ground_truth:
                if user_id not in recommendations:
                    continue
                
                rec_list = recommendations[user_id][:k]
                relevant_items = set(ground_truth[user_id])
                
                # Precision@K
                precision = self._precision_at_k(rec_list, relevant_items)
                precision_scores.append(precision)
                
                # Recall@K
                recall = self._recall_at_k(rec_list, relevant_items)
                recall_scores.append(recall)
                
                # F1@K
                f1 = self._f1_score(precision, recall)
                f1_scores.append(f1)
                
                # Average Precision
                ap = self._average_precision(recommendations[user_id], relevant_items)
                ap_scores.append(ap)
                
                # Reciprocal Rank
                rr = self._reciprocal_rank(recommendations[user_id], relevant_items)
                rr_scores.append(rr)
                
                # NDCG@K
                ndcg = self._ndcg_at_k(rec_list, relevant_items, k)
                ndcg_scores.append(ndcg)
            
            # Store average metrics
            results[f'Precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'Recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            results[f'F1@{k}'] = np.mean(f1_scores) if f1_scores else 0.0
            results[f'NDCG@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # Calculate MAP and MRR (not dependent on k)
        results['MAP'] = np.mean(ap_scores) if ap_scores else 0.0
        results['MRR'] = np.mean(rr_scores) if rr_scores else 0.0
        
        # Beyond-accuracy metrics
        if catalog_size:
            results['Coverage'] = self._catalog_coverage(recommendations, catalog_size)
        
        results['Diversity'] = self._average_diversity(recommendations)
        
        return results
    
    def evaluate_rankings(self, *args, **kwargs):
        """Alias for evaluate_recommendations for backward compatibility."""
        return self.evaluate_recommendations(*args, **kwargs)
    
    def evaluate_ratings(self, 
                        predicted_ratings: np.ndarray, 
                        true_ratings: np.ndarray) -> Dict[str, float]:
        """
        Evaluate rating predictions.
        
        Args:
            predicted_ratings: Predicted rating values
            true_ratings: True rating values
            
        Returns:
            Dictionary with RMSE and MAE
        """
        rmse = np.sqrt(np.mean((predicted_ratings - true_ratings) ** 2))
        mae = np.mean(np.abs(predicted_ratings - true_ratings))
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def _precision_at_k(self, recommendations: List[int], relevant_items: Set[int]) -> float:
        """Calculate Precision@K."""
        if not recommendations:
            return 0.0
        
        relevant_in_rec = len(set(recommendations) & relevant_items)
        return relevant_in_rec / len(recommendations)
    
    def _recall_at_k(self, recommendations: List[int], relevant_items: Set[int]) -> float:
        """Calculate Recall@K."""
        if not relevant_items:
            return 0.0
        
        relevant_in_rec = len(set(recommendations) & relevant_items)
        return relevant_in_rec / len(relevant_items)
    
    def _f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _average_precision(self, recommendations: List[int], relevant_items: Set[int]) -> float:
        """
        Calculate Average Precision.
        
        AP = Σ(P(k) × rel(k)) / |relevant_items|
        where P(k) is precision at position k, rel(k) is 1 if item at k is relevant
        """
        if not relevant_items:
            return 0.0
        
        score = 0.0
        num_hits = 0
        
        for i, item in enumerate(recommendations):
            if item in relevant_items:
                num_hits += 1
                score += num_hits / (i + 1)
        
        return score / len(relevant_items)
    
    def _reciprocal_rank(self, recommendations: List[int], relevant_items: Set[int]) -> float:
        """
        Calculate Reciprocal Rank.
        
        RR = 1 / rank_of_first_relevant_item
        """
        for i, item in enumerate(recommendations):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def _ndcg_at_k(self, recommendations: List[int], relevant_items: Set[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG@K = DCG@K / IDCG@K
        where DCG@K = Σ(2^rel_i - 1) / log2(i + 1)
        """
        def dcg(scores: List[float], k: int) -> float:
            """Calculate DCG for binary relevance (0 or 1)."""
            scores = scores[:k]
            if not scores:
                return 0.0
            return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(scores))
        
        # Create relevance scores (1 if relevant, 0 otherwise)
        relevance = [1 if item in relevant_items else 0 for item in recommendations[:k]]
        
        # Calculate DCG
        dcg_score = dcg(relevance, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(relevant_items), k)
        ideal_relevance.extend([0] * (k - len(ideal_relevance)))
        idcg_score = dcg(ideal_relevance, k)
        
        # Calculate NDCG
        if idcg_score == 0:
            return 0.0
        return dcg_score / idcg_score
    
    def _catalog_coverage(self, recommendations: Dict[int, List[int]], catalog_size: int) -> float:
        """
        Calculate catalog coverage.
        
        Coverage = |unique_recommended_items| / |total_items|
        """
        recommended_items = set()
        for rec_list in recommendations.values():
            recommended_items.update(rec_list)
        
        return len(recommended_items) / catalog_size
    
    def _average_diversity(self, recommendations: Dict[int, List[int]], 
                          similarity_matrix: np.ndarray = None) -> float:
        """
        Calculate average intra-list diversity.
        
        If similarity_matrix is not provided, returns the ratio of unique items
        in each recommendation list (simple diversity).
        """
        if similarity_matrix is not None:
            # Calculate diversity based on item similarities
            diversity_scores = []
            for user_id, rec_list in recommendations.items():
                if len(rec_list) < 2:
                    continue
                
                # Calculate average pairwise distance
                distances = []
                for i in range(len(rec_list)):
                    for j in range(i + 1, len(rec_list)):
                        # Distance = 1 - similarity
                        dist = 1 - similarity_matrix[rec_list[i], rec_list[j]]
                        distances.append(dist)
                
                if distances:
                    diversity_scores.append(np.mean(distances))
            
            return np.mean(diversity_scores) if diversity_scores else 0.0
        else:
            # Simple diversity: ratio of unique items
            diversity_scores = []
            for rec_list in recommendations.values():
                if rec_list:
                    diversity = len(set(rec_list)) / len(rec_list)
                    diversity_scores.append(diversity)
            
            return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def cross_validate(self, 
                      data: pd.DataFrame,
                      model,
                      n_folds: int = 5,
                      random_state: int = 42) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            data: DataFrame with columns ['user_id', 'item_id', 'rating'] (optional: 'timestamp')
            model: Recommender model with fit() and recommend() methods
            n_folds: Number of folds
            random_state: Random seed
            
        Returns:
            Dictionary mapping metric names to lists of scores per fold
        """
        from sklearn.model_selection import KFold
        
        # Create user-item matrix for splitting
        users = data['user_id'].unique()
        np.random.seed(random_state)
        np.random.shuffle(users)
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_results = defaultdict(list)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(users)):
            print(f"Evaluating fold {fold + 1}/{n_folds}")
            
            # Split users
            train_users = users[train_idx]
            test_users = users[test_idx]
            
            # Split data
            train_data = data[data['user_id'].isin(train_users)]
            test_data = data[data['user_id'].isin(test_users)]
            
            # Train model
            model.fit(train_data)
            
            # Generate recommendations
            recommendations = {}
            ground_truth = {}
            
            for user_id in test_users:
                user_items = test_data[test_data['user_id'] == user_id]['item_id'].tolist()
                if user_items:
                    # Get recommendations
                    recs = model.recommend(user_id, n_items=max(self.k_values))
                    recommendations[user_id] = recs
                    ground_truth[user_id] = user_items
            
            # Evaluate
            metrics = self.evaluate_recommendations(recommendations, ground_truth)
            
            # Store results
            for metric, value in metrics.items():
                fold_results[metric].append(value)
        
        return dict(fold_results)
    
    def print_results(self, results: Dict[str, float], decimal_places: int = 4):
        """Pretty print evaluation results."""
        print("\n" + "="*50)
        print("RECOMMENDATION SYSTEM EVALUATION RESULTS")
        print("="*50)
        
        # Group metrics by type
        ranking_metrics = ['Precision', 'Recall', 'F1', 'NDCG']
        overall_metrics = ['MAP', 'MRR']
        beyond_accuracy = ['Coverage', 'Diversity']
        error_metrics = ['RMSE', 'MAE']
        
        # Print ranking metrics
        print("\nRanking Metrics:")
        for metric_type in ranking_metrics:
            values = [(k, v) for k, v in results.items() if metric_type in k]
            if values:
                print(f"  {metric_type}:")
                for k, v in sorted(values):
                    print(f"    {k}: {v:.{decimal_places}f}")
        
        # Print overall metrics
        print("\nOverall Ranking Metrics:")
        for metric in overall_metrics:
            if metric in results:
                print(f"  {metric}: {results[metric]:.{decimal_places}f}")
        
        # Print beyond-accuracy metrics
        print("\nBeyond-Accuracy Metrics:")
        for metric in beyond_accuracy:
            if metric in results:
                print(f"  {metric}: {results[metric]:.{decimal_places}f}")
        
        # Print error metrics if available
        if any(metric in results for metric in error_metrics):
            print("\nRating Prediction Metrics:")
            for metric in error_metrics:
                if metric in results:
                    print(f"  {metric}: {results[metric]:.{decimal_places}f}")
        
        print("="*50 + "\n")

