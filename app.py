import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import json
from datetime import datetime

# Try to load spaCy, with fallback
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model (one-time)...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=False)
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"spaCy error: {e}")
    nlp = None

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
if 'product_embeddings' not in st.session_state:
    st.session_state.product_embeddings = None
if 'client' not in st.session_state:
    st.session_state.client = None

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
    1. **Upload product data** (CSV)
    2. **Upload user history** (products bought)
    3. **Set preferences**
    4. **Get recommendations**
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["Setup Data", "Get Recommendations", "Analysis"])

# TAB 1: DATA SETUP
with tab1:
    st.header("1️⃣ Upload Product Data")
    
    data_option = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload products CSV", type="csv")
        if uploaded_file is not None:
            st.session_state.products_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(st.session_state.products_df)} products")
            st.dataframe(st.session_state.products_df.head())
    
    else:
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
        st.success(f"✓ Using sample data")
        st.dataframe(st.session_state.products_df)
    
    if st.session_state.products_df is not None and nlp is not None:
        if st.button("🚀 Generate Product Embeddings", use_container_width=True, type="primary"):
            with st.spinner("Generating embeddings..."):
                descriptions = st.session_state.products_df['description'].tolist()
                embeddings = []
                
                for desc in descriptions:
                    doc = nlp(desc)
                    embeddings.append(doc.vector)
                
                st.session_state.product_embeddings = np.array(embeddings)
                st.success(f"✓ Generated embeddings for {len(embeddings)} products")

# TAB 2: GET RECOMMENDATIONS
with tab2:
    st.header("2️⃣ Get Recommendations")
    
    if st.session_state.products_df is None:
        st.warning("Please upload product data first")
    elif st.session_state.product_embeddings is None:
        st.warning("Please generate embeddings first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            user_pref_weight = st.slider("User Preference Weight (%)", 0, 100, 70) / 100
        
        with col2:
            use_margin = st.checkbox("Use Profit Margin", value=True)
            use_inventory = st.checkbox("Use Inventory Level", value=True)
        
        st.markdown("---")
        
        history_option = st.radio("User history:", ["Select Products", "Paste Product IDs"])
        user_product_ids = []
        
        if history_option == "Select Products":
            selected = st.multiselect("Select purchased products:", st.session_state.products_df['product_name'].tolist())
            if selected:
                user_product_ids = st.session_state.products_df[st.session_state.products_df['product_name'].isin(selected)]['product_id'].tolist()
        else:
            ids_input = st.text_area("Paste product IDs (comma-separated):", placeholder="P1, P3, P5")
            if ids_input:
                user_product_ids = [id.strip() for id in ids_input.split(",")]
        
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        if st.button("📊 Get Recommendations", use_container_width=True, type="primary"):
            if not user_product_ids:
                st.error("Please select or input user history")
            else:
                with st.spinner("Calculating..."):
                    user_products_mask = st.session_state.products_df['product_id'].isin(user_product_ids)
                    user_product_indices = user_products_mask.to_numpy().nonzero()[0]
                    
                    user_embeddings = st.session_state.product_embeddings[user_product_indices]
                    user_pref_vector = np.mean(user_embeddings, axis=0)
                    
                    similarities = cosine_similarity([user_pref_vector], st.session_state.product_embeddings)[0]
                    
                    all_scores = []
                    
                    for idx, (prod_id, similarity) in enumerate(zip(st.session_state.products_df['product_id'], similarities)):
                        if prod_id in user_product_ids:
                            continue
                        
                        business_weight = 0
                        if use_margin:
                            business_weight += st.session_state.products_df.iloc[idx]['margin']
                        if use_inventory:
                            inv_level = st.session_state.products_df.iloc[idx]['inventory']
                            business_weight += min(inv_level / 100, 0.5)
                        
                        final_score = similarity * (1 + (business_weight * (1 - user_pref_weight)))
                        
                        all_scores.append({
                            'product_id': prod_id,
                            'product_name': st.session_state.products_df.iloc[idx]['product_name'],
                            'description': st.session_state.products_df.iloc[idx]['description'],
                            'similarity': similarity,
                            'final_score': final_score,
                            'margin': st.session_state.products_df.iloc[idx]['margin'],
                            'inventory': st.session_state.products_df.iloc[idx]['inventory']
                        })
                    
                    recommendations = sorted(all_scores, key=lambda x: x['final_score'], reverse=True)
                    top_recommendations = recommendations[:num_recommendations]
                    
                    st.subheader("✨ Top Recommendations")
                    
                    for rank, rec in enumerate(top_recommendations, 1):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {rank}. {rec['product_name']}")
                            st.caption(rec['description'])
                        
                        with col2:
                            st.metric("Match", f"{rec['similarity']:.1%}")
                            st.metric("Score", f"{rec['final_score']:.2f}")
                        
                        with col3:
                            st.metric("Margin", f"{rec['margin']:.0%}")
                            st.metric("Inventory", f"{rec['inventory']}")
                        
                        st.markdown("---")
                    
                    if st.session_state.client and st.button("🤖 Get AI Explanations"):
                        with st.spinner("Claude analyzing..."):
                            try:
                                recs_text = "\n".join([f"{i+1}. {r['product_name']}\n   {r['description']}" for i, r in enumerate(top_recommendations)])
                                message = st.session_state.client.messages.create(
                                    model="claude-opus-4-6",
                                    max_tokens=800,
                                    messages=[{"role": "user", "content": f"Why would these products be great for someone who bought {', '.join(user_product_ids)}?\n\n{recs_text}"}]
                                )
                                st.markdown("### 💡 AI Analysis")
                                st.markdown(message.content[0].text)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

# TAB 3: ANALYSIS
with tab3:
    st.header("3️⃣ Analysis")
    
    if st.session_state.products_df is None:
        st.warning("Upload product data first")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Products", len(st.session_state.products_df))
        with col2:
            st.metric("Avg Margin", f"{st.session_state.products_df['margin'].mean():.1%}")
        with col3:
            st.metric("Total Inventory", int(st.session_state.products_df['inventory'].sum()))
        
        st.markdown("---")
        st.subheader("Product Data")
        st.dataframe(st.session_state.products_df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit + spaCy + Claude API")