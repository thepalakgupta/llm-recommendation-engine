"""Unit tests for TF-IDF Vectorizer."""

import pytest
import numpy as np
from src.vectorizer import ProductVectorizer

class TestProductVectorizer:
    """Test cases for product vectorizer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.descriptions = [
            "Durable travel backpack perfect for hiking",
            "Compression bags for travel packing",
            "Coffee maker with timer",
            "Waterproof hiking boots"
        ]
        self.names = ["Backpack", "Bags", "Coffee", "Boots"]
        self.vectorizer = ProductVectorizer()
    
    def test_fit_transform(self):
        """Test fit and transform."""
        vectors = self.vectorizer.fit_transform(self.descriptions, self.names)
        
        assert vectors.shape[0] == 4  # 4 products
        assert vectors.shape[1] > 0   # Has features
        assert self.vectorizer.fitted
    
    def test_transform_after_fit(self):
        """Test transform after fitting."""
        self.vectorizer.fit_transform(self.descriptions, self.names)
        
        new_desc = ["travel backpack for hiking"]
        vectors = self.vectorizer.transform(new_desc)
        
        assert vectors.shape[0] == 1
        assert vectors.shape[1] == self.vectorizer.get_vocabulary_size()
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        with pytest.raises(ValueError):
            self.vectorizer.transform(self.descriptions)
    
    def test_get_vector_for_product(self):
        """Test getting vector for specific product."""
        self.vectorizer.fit_transform(self.descriptions, self.names)
        
        vector = self.vectorizer.get_vector_for_product("Backpack")
        assert len(vector) == self.vectorizer.get_vocabulary_size()
    
    def test_get_vector_nonexistent_product(self):
        """Test error for nonexistent product."""
        self.vectorizer.fit_transform(self.descriptions, self.names)
        
        with pytest.raises(ValueError):
            self.vectorizer.get_vector_for_product("NonExistent")
    
    def test_mismatched_lengths(self):
        """Test error for mismatched descriptions and names."""
        with pytest.raises(ValueError):
            self.vectorizer.fit_transform(self.descriptions, ["Only", "Two"])
    
    def test_vocabulary_size(self):
        """Test vocabulary size is reasonable."""
        self.vectorizer.fit_transform(self.descriptions, self.names)
        
        vocab_size = self.vectorizer.get_vocabulary_size()
        assert vocab_size > 0
        assert vocab_size < 100  # Should be reasonable