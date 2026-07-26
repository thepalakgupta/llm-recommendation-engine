"""
TF-IDF Vectorizer for product descriptions.

Converts text descriptions into numerical vectors
that can be compared using cosine similarity.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.config import TFIDF_CONFIG

class ProductVectorizer:
    """Convert product descriptions to TF-IDF vectors."""
    
    def __init__(self, config=None):
        """
        Initialize vectorizer with configuration.
        
        Args:
            config (dict): TF-IDF configuration parameters
        """
        self.config = config or TFIDF_CONFIG
        self.vectorizer = TfidfVectorizer(**self.config)
        self.fitted = False
        self.product_names = []
        self.vectors = None
    
    def fit_transform(self, descriptions, product_names):
        """
        Fit vectorizer on descriptions and transform to vectors.
        
        Args:
            descriptions (list): List of product descriptions
            product_names (list): List of product names (for reference)
            
        Returns:
            np.ndarray: TF-IDF vectors (n_products × n_features)
        """
        if len(descriptions) != len(product_names):
            raise ValueError("Number of descriptions must match number of names")
        
        self.vectors = self.vectorizer.fit_transform(descriptions)
        self.product_names = product_names
        self.fitted = True
        
        return self.vectors.toarray()
    
    def transform(self, descriptions):
        """
        Transform new descriptions using fitted vectorizer.
        
        Args:
            descriptions (list): List of product descriptions
            
        Returns:
            np.ndarray: TF-IDF vectors
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        vectors = self.vectorizer.transform(descriptions)
        return vectors.toarray()
    
    def get_feature_names(self):
        """
        Get list of feature names (words/phrases).
        
        Returns:
            list: Feature names from vectorizer vocabulary
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_vector_for_product(self, product_name):
        """
        Get TF-IDF vector for specific product.
        
        Args:
            product_name (str): Name of product
            
        Returns:
            np.ndarray: TF-IDF vector for product
        """
        if product_name not in self.product_names:
            raise ValueError(f"Product '{product_name}' not found in vectorizer")
        
        idx = self.product_names.index(product_name)
        return self.vectors[idx].toarray().flatten()
    
    def get_vocabulary_size(self):
        """Get number of features in vocabulary."""
        return len(self.vectorizer.get_feature_names_out())
    
    def get_top_features_for_product(self, product_name, n=10):
        """
        Get top features (words) for a specific product.
        
        Args:
            product_name (str): Product name
            n (int): Number of top features to return
            
        Returns:
            list: Top feature names and their scores
        """
        if product_name not in self.product_names:
            raise ValueError(f"Product '{product_name}' not found")
        
        idx = self.product_names.index(product_name)
        vector = self.vectors[idx].toarray().flatten()
        feature_names = self.get_feature_names()
        
        # Get top n indices
        top_indices = np.argsort(vector)[-n:][::-1]
        
        return [(feature_names[i], vector[i]) for i in top_indices if vector[i] > 0]