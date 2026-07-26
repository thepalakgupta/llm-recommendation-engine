"""Utility functions for recommendation engine."""

import pandas as pd
import json

def load_products_from_csv(filepath):
    """
    Load product catalog from CSV file.
    
    Expected columns: product_id, name, description, category
    """
    df = pd.read_csv(filepath)
    return df

def load_user_history_from_json(filepath):
    """Load user purchase history from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_recommendations_to_csv(recommendations, filepath):
    """Save recommendations to CSV file."""
    df = pd.DataFrame(recommendations)
    df.to_csv(filepath, index=False)
    print(f"Recommendations saved to {filepath}")

def print_recommendations(recommendations):
    """Pretty print recommendations."""
    print("\n" + "="*70)
    print("PRODUCT RECOMMENDATIONS")
    print("="*70)
    print(f"{'Rank':<6} {'Product':<30} {'Similarity':<15}")
    print("-"*70)
    
    for rec in recommendations:
        print(f"{rec['rank']:<6} {rec['product_name']:<30} "
              f"{rec['similarity_score']:<15.4f}")
    
    print("="*70 + "\n")

def print_user_recommendations(user_id, recommendations):
    """Print recommendations for specific user."""
    print(f"\nRecommendations for User {user_id}:")
    print("-" * 50)
    
    for rec in recommendations:
        print(f"{rec['rank']}. {rec['product_name']} "
              f"(Similarity: {rec['similarity_score']:.2%})")

def create_sample_data():
    """Create sample products and user data for testing."""
    products = {
        'products': [
            {
                'id': 1,
                'name': 'Travel Backpack',
                'description': 'Durable travel backpack perfect for hiking and outdoor adventures',
                'category': 'Travel Gear'
            },
            {
                'id': 2,
                'name': 'Compression Bags',
                'description': 'Lightweight compression bags for travel packing',
                'category': 'Travel Gear'
            },
            {
                'id': 3,
                'name': 'Coffee Maker',
                'description': 'Electric coffee maker with programmable timer',
                'category': 'Home & Kitchen'
            },
            {
                'id': 4,
                'name': 'Hiking Boots',
                'description': 'Waterproof hiking boots for outdoor trails and mountains',
                'category': 'Sports & Outdoors'
            },
            {
                'id': 5,
                'name': 'Travel Pillow',
                'description': 'Comfort travel pillow with cooling gel for flights',
                'category': 'Travel Gear'
            },
            {
                'id': 6,
                'name': 'Bluetooth Speaker',
                'description': 'Portable Bluetooth speaker for outdoor activities',
                'category': 'Electronics'
            },
            {
                'id': 7,
                'name': 'Tent',
                'description': 'Waterproof camping tent for outdoor expeditions',
                'category': 'Sports & Outdoors'
            },
            {
                'id': 8,
                'name': 'Power Bank',
                'description': 'High capacity power bank for charging on the go',
                'category': 'Electronics'
            },
            {
                'id': 9,
                'name': 'Water Bottle',
                'description': 'Insulated water bottle for travel and outdoor activities',
                'category': 'Travel Gear'
            },
            {
                'id': 10,
                'name': 'Laptop Backpack',
                'description': 'Professional laptop backpack for work and travel',
                'category': 'Travel Gear'
            }
        ]
    }
    
    return products