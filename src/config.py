"""Configuration for LLM Recommendation Engine."""

# TF-IDF Configuration
TFIDF_CONFIG = {
    'max_features': 1000,
    'max_df': 0.95,  # Ignore terms that appear in >95% of documents
    'min_df': 1,     # Ignore terms that appear in <1 document
    'ngram_range': (1, 2),  # Unigrams and bigrams
    'lowercase': True,
    'stop_words': 'english'
}

# Recommendation Configuration
RECOMMENDATION_CONFIG = {
    'top_n': 5,  # Return top 5 recommendations
    'min_similarity': 0.0,  # Minimum similarity threshold
    'exclude_purchased': True  # Don't recommend already purchased items
}

# Product Categories
PRODUCT_CATEGORIES = [
    'Travel Gear',
    'Electronics',
    'Home & Kitchen',
    'Sports & Outdoors',
    'Fashion'
]

# Similarity Score Interpretation
SIMILARITY_SCALE = {
    (0.9, 1.0): 'Highly Similar',
    (0.7, 0.9): 'Very Similar',
    (0.5, 0.7): 'Similar',
    (0.3, 0.5): 'Somewhat Similar',
    (0.0, 0.3): 'Low Similarity'
}