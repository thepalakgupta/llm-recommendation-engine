import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
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
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'product_vectors' not in st.session_state:
    st.session_state.product_vectors = None
if 'client' not in st.session_state:
    st.session_state.client = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.text_input("Claude API Key", type="password")
    if api_key:
        st.session_state.client = anthropic.Anthropic(api_key=api_key)
        st.success("✓ Claude Connected")

# Tabs
tab1, tab2, tab3 = st.tabs(["Setup Data", "Get Recommendations", "Analysis"])

# TAB 1: SETUP
with tab1:
    st.header("1️⃣ Upload Product Data")
    
    data_option = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV"])
    
    if data_option == "Use Sample Data":
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
    
    else:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            st.session_state.products_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(st.session_state.products_df)} products")
            st.dataframe(st.session_state.products_df)
    
    if st.session_state.products_df is not None:
        if st.button("🚀 Generate Vectors", use_container_width=True, type="primary"):
            with st.spinner("Generating vectors..."):
                descriptions = st.session_state.products_df['description'].tolist()
                
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                product_vectors = vectorizer.fit_transform(descriptions).toarray()
                
                st.session_state.vectorizer = vectorizer
                st.session_state.product_vectors = product_vectors
                
                st.success(f"✓ Generated vectors for {len(product_vectors)} products")
                st.info(f"Vector size: {product_vectors.shape[1]} dimensions")

# TAB 2: RECOMMENDATIONS
with tab2:
    st.header("2️⃣ Get Recommendations")
    
    if st.session_state.products_df is None:
        st.warning("Upload data first")
    elif st.session_state.product_vectors is None:
        st.warning("Generate vectors first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            user_pref_weight = st.slider("User Preference Weight (%)", 0, 100, 70) / 100
        
        with col2:
            use_margin = st.checkbox("Use Margin", value=True)
            use_inventory = st.checkbox("Use Inventory", value=True)
        
        st.markdown("---")
        
        history_option = st.radio("User history:", ["Select", "Paste IDs"])
        user_product_ids = []
        
        if history_option == "Select":
            selected = st.multiselect("Products purchased:", st.session_state.products_df['product_name'].tolist())
            if selected:
                user_product_ids = st.session_state.products_df[st.session_state.products_df['product_name'].isin(selected)]['product_id'].tolist()
        else:
            ids_input = st.text_area("Paste IDs (P1, P2, P3):")
            if ids_input:
                user_product_ids = [id.strip() for id in ids_input.split(",")]
        
        num_recs = st.slider("How many?", 1, 10, 5)
        
        if st.button("📊 Get Recommendations", use_container_width=True, type="primary"):
            if not user_product_ids:
                st.error("Select or paste user history")
            else:
                with st.spinner("Calculating..."):
                    # Get user vectors
                    user_mask = st.session_state.products_df['product_id'].isin(user_product_ids)
                    user_indices = user_mask.to_numpy().nonzero()[0]
                    
                    user_vectors = st.session_state.product_vectors[user_indices]
                    user_avg_vector = np.mean(user_vectors, axis=0).reshape(1, -1)
                    
                    # Calculate similarities
                    similarities = cosine_similarity(user_avg_vector, st.session_state.product_vectors)[0]
                    
                    # Score recommendations
                    all_scores = []
                    
                    for idx, (prod_id, sim) in enumerate(zip(st.session_state.products_df['product_id'], similarities)):
                        if prod_id in user_product_ids:
                            continue
                        
                        biz_weight = 0
                        if use_margin:
                            biz_weight += st.session_state.products_df.iloc[idx]['margin']
                        if use_inventory:
                            inv = st.session_state.products_df.iloc[idx]['inventory']
                            biz_weight += min(inv / 100, 0.5)
                        
                        final_score = sim * (1 + (biz_weight * (1 - user_pref_weight)))
                        
                        all_scores.append({
                            'product_id': prod_id,
                            'product_name': st.session_state.products_df.iloc[idx]['product_name'],
                            'description': st.session_state.products_df.iloc[idx]['description'],
                            'similarity': sim,
                            'final_score': final_score,
                            'margin': st.session_state.products_df.iloc[idx]['margin'],
                            'inventory': st.session_state.products_df.iloc[idx]['inventory']
                        })
                    
                    recs = sorted(all_scores, key=lambda x: x['final_score'], reverse=True)[:num_recs]
                    
                    st.subheader("✨ Top Recommendations")
                    
                    for rank, rec in enumerate(recs, 1):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {rank}. {rec['product_name']}")
                            st.caption(rec['description'])
                        
                        with col2:
                            st.metric("Match", f"{rec['similarity']:.1%}")
                            st.metric("Score", f"{rec['final_score']:.2f}")
                        
                        with col3:
                            st.metric("Margin", f"{rec['margin']:.0%}")
                            st.metric("Stock", rec['inventory'])
                        
                        st.markdown("---")
                    
                    if st.session_state.client:
                        if st.button("🤖 Get AI Explanations"):
                            with st.spinner("Claude thinking..."):
                                try:
                                    recs_text = "\n".join([f"{i+1}. {r['product_name']}" for i, r in enumerate(recs)])
                                    msg = st.session_state.client.messages.create(
                                        model="claude-opus-4-6",
                                        max_tokens=600,
                                        messages=[{"role": "user", "content": f"User bought: {', '.join(user_product_ids)}\n\nTop recommendations:\n{recs_text}\n\nWhy are these good recommendations? (2 sentences each)"}]
                                    )
                                    st.markdown("### 💡 Why These?")
                                    st.markdown(msg.content[0].text)
                                except Exception as e:
                                    st.error(f"Error: {e}")

# TAB 3: ANALYSIS
with tab3:
    st.header("3️⃣ Analysis")
    
    if st.session_state.products_df is None:
        st.warning("Upload data first")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Products", len(st.session_state.products_df))
        with col2:
            st.metric("Avg Margin", f"{st.session_state.products_df['margin'].mean():.1%}")
        with col3:
            st.metric("Total Stock", int(st.session_state.products_df['inventory'].sum()))
        
        st.markdown("---")
        st.dataframe(st.session_state.products_df, use_container_width=True)

st.caption("Streamlit + TF-IDF + Claude API")