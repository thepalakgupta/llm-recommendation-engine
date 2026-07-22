import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import json
from datetime import datetime

st.set_page_config(
    page_title="LLM Recommendation Engine",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 LLM-Powered Recommendation Engine")
st.markdown("Smart product recommendations balancing user preference + business metrics")

# Initialize session state
if 'products_df' not in st.session_state:
    st.session_state.products_df = None
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
if 'product_embeddings' not in st.session_state:
    st.session_state.product_embeddings = None
if 'client' not in st.session_state:
    st.session_state.client = None

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    return nlp

# Sidebar - API Key & Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.text_input("Claude API Key", type="password", help="Get from console.anthropic.com")
    if api_key:
        st.session_state.client = anthropic.Anthropic(api_key=api_key)
        st.success("✓ Claude API Connected")
    
    st.markdown("---")
    st.markdown("""
    ### How it works
    1. **Upload product data** (CSV with descriptions)
    2. **Upload user history** (products they bought)
    3. **Set preferences:**
       - User preference weight (0-100%)
       - Business metrics to use
       - Number of recommendations
    4. **Get smart recommendations** with Claude explanations
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["Setup Data", "Get Recommendations", "Analysis"])

# TAB 1: DATA SETUP
with tab1:
    st.header("1️⃣ Upload Product Data")
    
    # Load spaCy model
    with st.spinner("Loading spaCy model..."):
        st.session_state.nlp = load_spacy_model()
    
    # Option to use sample data or upload
    data_option = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload products CSV", type="csv")
        
        if uploaded_file is not None:
            st.session_state.products_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(st.session_state.products_df)} products")
            
            st.subheader("Sample of your data:")
            st.dataframe(st.session_state.products_df.head())
    
    else:
        # Sample data
        sample_data = {
            'product_id': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
            'product_name': [
                'Durable Travel Backpack',
                'Comfort Travel Pillow',
                'Waterproof Hiking Jacket',
                'Portable Phone Charger',
                'Luggage Organizer',
                'Hiking Boots',
                'Travel Socks Set',
                'Compression Packing Cubes'
            ],
            'description': [
                'Durable 40L waterproof backpack perfect for hiking and travel',
                'Memory foam travel pillow with cooling gel, great for flights',
                'Waterproof breathable jacket ideal for hiking and outdoor activities',
                'Fast charging 20000mAh portable battery for travel',
                'Packing organizer cubes for luggage and backpack',
                'Durable waterproof hiking boots with ankle support',
                'Merino wool travel socks, moisture-wicking',
                'Compression cubes to maximize packing space'
            ],
            'margin': [0.35, 0.45, 0.40, 0.50, 0.55, 0.30, 0.60, 0.48],
            'inventory': [50, 200, 30, 150, 100, 15, 300, 80],
            'popularity': [0.85, 0.92, 0.78, 0.88, 0.82, 0.80, 0.75, 0.70]
        }
        
        st.session_state.products_df = pd.DataFrame(sample_data)
        st.success(f"✓ Using sample data with {len(st.session_state.products_df)} products")
        
        st.subheader("Sample Products:")
        st.dataframe(st.session_state.products_df)
    
    # Generate embeddings
    if st.session_state.products_df is not None and st.session_state.nlp is not None:
        if st.button("🚀 Generate Product Embeddings", use_container_width=True, type="primary"):
            with st.spinner("Generating embeddings for products..."):
                descriptions = st.session_state.products_df['description'].tolist()
                embeddings = []
                
                for desc in descriptions:
                    doc = st.session_state.nlp(desc)
                    embeddings.append(doc.vector)
                
                st.session_state.product_embeddings = np.array(embeddings)
                st.success(f"✓ Generated embeddings for {len(embeddings)} products")
                st.info(f"Each product converted to {len(embeddings[0])} dimensions using spaCy")

# TAB 2: GET RECOMMENDATIONS
with tab2:
    st.header("2️⃣ Get Recommendations")
    
    if st.session_state.products_df is None:
        st.warning("Please upload product data first (go to Setup Data tab)")
    
    elif st.session_state.product_embeddings is None:
        st.warning("Please generate embeddings first (go to Setup Data tab)")
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Preferences")
            
            user_pref_weight = st.slider(
                "User Preference Weight (%)",
                0, 100, 70,
                help="How much to prioritize user preference vs business metrics"
            ) / 100
            
            st.caption(f"User: {user_pref_weight*100:.0f}% | Business: {(1-user_pref_weight)*100:.0f}%")
        
        with col2:
            st.subheader("Business Metrics")
            
            use_margin = st.checkbox("Use Profit Margin", value=True)
            use_inventory = st.checkbox("Use Inventory Level", value=True)
            use_popularity = st.checkbox("Use Popularity", value=False)
        
        st.markdown("---")
        
        st.subheader("User History")
        
        # Option: Select products or paste list
        history_option = st.radio("How to input user history:", ["Select Products", "Paste Product IDs"])
        
        user_product_ids = []
        
        if history_option == "Select Products":
            product_options = st.session_state.products_df['product_name'].tolist()
            selected_products = st.multiselect(
                "Select products user has purchased:",
                product_options
            )
            
            if selected_products:
                user_product_ids = st.session_state.products_df[
                    st.session_state.products_df['product_name'].isin(selected_products)
                ]['product_id'].tolist()
        
        else:
            product_ids_input = st.text_area(
                "Paste product IDs (comma-separated):",
                placeholder="P1, P3, P5"
            )
            if product_ids_input:
                user_product_ids = [id.strip() for id in product_ids_input.split(",")]
        
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        if st.button("📊 Get Recommendations", use_container_width=True, type="primary"):
            if not user_product_ids:
                st.error("Please select or input user product history")
            else:
                with st.spinner("Calculating recommendations..."):
                    # Get user's purchased products
                    user_products_mask = st.session_state.products_df['product_id'].isin(user_product_ids)
                    user_product_indices = user_products_mask.to_numpy().nonzero()[0]
                    
                    # Create user preference vector (average of purchased products)
                    user_embeddings = st.session_state.product_embeddings[user_product_indices]
                    user_pref_vector = np.mean(user_embeddings, axis=0)
                    
                    # Calculate similarity scores
                    similarities = cosine_similarity([user_pref_vector], st.session_state.product_embeddings)[0]
                    
                    # Prepare recommendation scores
                    all_scores = []
                    
                    for idx, (prod_id, similarity) in enumerate(zip(
                        st.session_state.products_df['product_id'],
                        similarities
                    )):
                        # Skip products already purchased
                        if prod_id in user_product_ids:
                            continue
                        
                        # Calculate business weight
                        business_weight = 0
                        if use_margin:
                            business_weight += st.session_state.products_df.iloc[idx]['margin']
                        if use_inventory:
                            inv_level = st.session_state.products_df.iloc[idx]['inventory']
                            business_weight += min(inv_level / 100, 0.5)
                        if use_popularity:
                            business_weight += st.session_state.products_df.iloc[idx]['popularity']
                        
                        # Final score: similarity boosted by business metrics
                        final_score = similarity * (1 + (business_weight * (1 - user_pref_weight)))
                        
                        all_scores.append({
                            'index': idx,
                            'product_id': prod_id,
                            'product_name': st.session_state.products_df.iloc[idx]['product_name'],
                            'description': st.session_state.products_df.iloc[idx]['description'],
                            'similarity': similarity,
                            'business_weight': business_weight,
                            'final_score': final_score,
                            'margin': st.session_state.products_df.iloc[idx]['margin'],
                            'inventory': st.session_state.products_df.iloc[idx]['inventory'],
                            'popularity': st.session_state.products_df.iloc[idx]['popularity']
                        })
                    
                    # Sort by final score
                    recommendations = sorted(all_scores, key=lambda x: x['final_score'], reverse=True)
                    top_recommendations = recommendations[:num_recommendations]
                    
                    # Display recommendations
                    st.subheader("✨ Top Recommendations")
                    
                    for rank, rec in enumerate(top_recommendations, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"### {rank}. {rec['product_name']}")
                                st.caption(rec['description'])
                            
                            with col2:
                                st.metric("Match Score", f"{rec['similarity']:.2%}")
                                st.metric("Final Score", f"{rec['final_score']:.2f}")
                            
                            with col3:
                                st.metric("Margin", f"{rec['margin']:.0%}")
                                st.metric("Inventory", f"{rec['inventory']} units")
                            
                            st.markdown("---")
                    
                    # Get Claude AI explanations
                    if st.session_state.client and st.button("🤖 Get AI Explanations", use_container_width=True):
                        with st.spinner("Claude is analyzing recommendations..."):
                            try:
                                recommendations_text = "\n".join([
                                    f"{i+1}. {rec['product_name']} (Score: {rec['final_score']:.2f})\n   {rec['description']}"
                                    for i, rec in enumerate(top_recommendations)
                                ])
                                
                                message = st.session_state.client.messages.create(
                                    model="claude-opus-4-6",
                                    max_tokens=1000,
                                    messages=[{
                                        "role": "user",
                                        "content": f"""You're a product recommendation expert. Analyze these recommendations for a user 
who has previously purchased: {', '.join(user_product_ids)}

Top recommendations:
{recommendations_text}

For each recommendation, briefly explain (2-3 sentences):
- Why it matches the user's taste
- What problem it solves
- How it complements their previous purchases

Be concise and actionable."""
                                    }]
                                )
                                
                                st.markdown("### 💡 AI Analysis")
                                st.markdown(message.content[0].text)
                            
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.info("Enter Claude API key in sidebar to get AI explanations")
                    
                    # Export option
                    st.markdown("---")
                    
                    export_df = pd.DataFrame([
                        {
                            'Rank': i+1,
                            'Product': rec['product_name'],
                            'Match %': f"{rec['similarity']:.1%}",
                            'Score': f"{rec['final_score']:.2f}",
                            'Margin': f"{rec['margin']:.0%}",
                            'Inventory': rec['inventory']
                        }
                        for i, rec in enumerate(top_recommendations)
                    ])
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Recommendations",
                        csv,
                        f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )

# TAB 3: ANALYSIS
with tab3:
    st.header("3️⃣ Analysis & Metrics")
    
    if st.session_state.products_df is None:
        st.warning("Please upload product data first")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(st.session_state.products_df))
        
        with col2:
            avg_margin = st.session_state.products_df['margin'].mean()
            st.metric("Avg Margin", f"{avg_margin:.1%}")
        
        with col3:
            total_inventory = st.session_state.products_df['inventory'].sum()
            st.metric("Total Inventory", int(total_inventory))
        
        with col4:
            avg_popularity = st.session_state.products_df['popularity'].mean()
            st.metric("Avg Popularity", f"{avg_popularity:.1%}")
        
        st.markdown("---")
        
        st.subheader("Product Metrics Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Profit Margin Distribution")
            margin_data = st.session_state.products_df[['product_name', 'margin']].sort_values('margin', ascending=True)
            
            import plotly.express as px
            fig1 = px.bar(margin_data, x='margin', y='product_name', orientation='h', title="Profit Margin by Product")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Inventory Levels")
            inv_data = st.session_state.products_df[['product_name', 'inventory']].sort_values('inventory', ascending=True)
            
            fig2 = px.bar(inv_data, x='inventory', y='product_name', orientation='h', title="Inventory by Product")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Full Product Data")
        st.dataframe(st.session_state.products_df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit + spaCy + Claude API | GitHub: thepalakgupta/llm-recommendation-engine")