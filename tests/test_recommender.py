"""Unit tests for Recommendation Engine."""

import pytest
from src.vectorizer import ProductVectorizer
from src.recommender import RecommendationEngine

class TestRecommendationEngine:
    """Test cases for recommendation engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.descriptions = [
            "Durable travel backpack perfect for hiking",
            "Compression bags for travel packing",
            "Coffee maker with timer",
            "Waterproof hiking boots",
            "Travel pillow for flights"
        ]
        self.names = ["Backpack", "Bags", "Coffee", "Boots", "Pillow"]
        
        self.vectorizer = ProductVectorizer()
        self.vectorizer.fit_transform(self.descriptions, self.names)
        self.engine = RecommendationEngine(self.vectorizer)
    
    def test_init_without_fitted_vectorizer(self):
        """Test that engine requires fitted vectorizer."""
        vec = ProductVectorizer()
        
        with pytest.raises(ValueError):
            RecommendationEngine(vec)
    
    def test_calculate_preference_vector(self):
        """Test preference vector calculation."""
        pref_vec = self.engine.calculate_user_preference_vector(["Backpack", "Boots"])
        
        assert len(pref_vec) == self.vectorizer.get_vocabulary_size()
    
    def test_preference_vector_empty_list(self):
        """Test error for empty purchase list."""
        with pytest.raises(ValueError):
            self.engine.calculate_user_preference_vector([])
    
    def test_preference_vector_nonexistent_product(self):
        """Test error for nonexistent product."""
        with pytest.raises(ValueError):
            self.engine.calculate_user_preference_vector(["NonExistent"])
    
    def test_get_recommendations(self):
        """Test getting recommendations."""
        recs = self.engine.get_recommendations(["Backpack"])
        
        assert len(recs) > 0
        assert len(recs) <= 5  # Default top_n
        assert all('product_name' in r for r in recs)
        assert all('similarity_score' in r for r in recs)
    
    def test_exclude_purchased(self):
        """Test that purchased products are excluded."""
        recs = self.engine.get_recommendations(
            ["Backpack"],
            exclude_purchased=True
        )
        
        product_names = [r['product_name'] for r in recs]
        assert "Backpack" not in product_names
    
    def test_top_n_parameter(self):
        """Test top_n parameter."""
        recs = self.engine.get_recommendations(["Backpack"], top_n=2)
        
        assert len(recs) <= 2
    
    def test_similarity_scores_in_range(self):
        """Test that similarity scores are in valid range."""
        recs = self.engine.get_recommendations(["Backpack"])
        
        for rec in recs:
            assert 0 <= rec['similarity_score'] <= 1
    
    def test_explain_recommendation(self):
        """Test recommendation explanation."""
        explanation = self.engine.explain_recommendation("Backpack")
        
        assert explanation['product_name'] == "Backpack"
        assert 'key_features' in explanation
        assert 'feature_scores' in explanation
    
    def test_batch_recommendations(self):
        """Test batch recommendations for multiple users."""
        user_purchases = {
            "user_1": ["Backpack"],
            "user_2": ["Coffee", "Pillow"]
        }
        
        results = self.engine.batch_recommendations(user_purchases)
        
        assert "user_1" in results
        assert "user_2" in results
        assert len(results["user_1"]) > 0