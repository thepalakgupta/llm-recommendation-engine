"""
Recommendation engine using cosine similarity.

Finds similar products based on user purchase history
and TF-IDF vectorized product descriptions.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.config import RECOMMENDATION_CONFIG

class RecommendationEngine:
    """Generate product recommendations using cosine similarity."""
    
    def __init__(self, vectorizer, config=None):
        """
        Initialize recommendation engine.
        
        Args:
            vectorizer (ProductVectorizer): Fitted vectorizer instance
            config (dict): Recommendation configuration
        """
        self.vectorizer = vectorizer
        self.config = config or RECOMMENDATION_CONFIG
        self.product_vectors = vectorizer.vectors
        self.product_names = vectorizer.product_names
        
        if self.product_vectors is None:
            raise ValueError("Vectorizer must be fitted before initializing engine")
    
    def calculate_user_preference_vector(self, purchased_product_names):
        """
        Calculate average preference vector from purchased products.
        
        Args:
            purchased_product_names (list): Names of products user purchased
            
        Returns:
            np.ndarray: Average TF-IDF vector of purchased products
        """
        if not purchased_product_names:
            raise ValueError("At least one purchased product required")
        
        vectors = []
        for name in purchased_product_names:
            if name not in self.product_names:
                raise ValueError(f"Product '{name}' not found in catalog")
            
            idx = self.product_names.index(name)
            vectors.append(self.product_vectors[idx])
        
        # Average of all vectors
        preference_vector = np.mean(vectors, axis=0)
        return preference_vector
    
    def get_recommendations(self, purchased_product_names, top_n=None, 
                           exclude_purchased=None):
        """
        Get product recommendations for a user.
        
        Args:
            purchased_product_names (list): Names of products user purchased
            top_n (int): Number of recommendations to return
            exclude_purchased (bool): Whether to exclude purchased products
            
        Returns:
            list: List of dicts with product name and similarity score
        """
        top_n = top_n or self.config['top_n']
        exclude_purchased = exclude_purchased if exclude_purchased is not None \
                           else self.config['exclude_purchased']
        
        # Get user preference vector
        preference_vector = self.calculate_user_preference_vector(
            purchased_product_names
        )
        
        # Calculate similarity with all products
        similarities = cosine_similarity(
            [preference_vector],
            self.product_vectors
        )[0]
        
        # Get top products
        recommendations = []
        sorted_indices = np.argsort(similarities)[::-1]
        
        for idx in sorted_indices:
            product_name = self.product_names[idx]
            similarity = similarities[idx]
            
            # Skip purchased products if configured
            if exclude_purchased and product_name in purchased_product_names:
                continue
            
            # Skip low similarity scores
            if similarity < self.config['min_similarity']:
                continue
            
            recommendations.append({
                'product_name': product_name,
                'similarity_score': round(float(similarity), 4),
                'rank': len(recommendations) + 1
            })
            
            if len(recommendations) >= top_n:
                break
        
        return recommendations
    
    def explain_recommendation(self, product_name):
        """
        Explain why a product is recommended.
        
        Args:
            product_name (str): Name of product
            
        Returns:
            dict: Explanation with top features
        """
        if product_name not in self.product_names:
            raise ValueError(f"Product '{product_name}' not found")
        
        # Get top features for this product
        top_features = self.vectorizer.get_top_features_for_product(
            product_name, 
            n=5
        )
        
        return {
            'product_name': product_name,
            'key_features': [f[0] for f in top_features],
            'feature_scores': [round(f[1], 4) for f in top_features]
        }
    
    def batch_recommendations(self, user_purchases_dict, top_n=None):
        """
        Get recommendations for multiple users.
        
        Args:
            user_purchases_dict (dict): {user_id: [purchased_products]}
            top_n (int): Number of recommendations per user
            
        Returns:
            dict: {user_id: [recommendations]}
        """
        results = {}
        
        for user_id, purchases in user_purchases_dict.items():
            try:
                results[user_id] = self.get_recommendations(purchases, top_n)
            except ValueError as e:
                results[user_id] = {'error': str(e)}
        
        return results