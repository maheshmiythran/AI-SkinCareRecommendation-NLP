import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="SmartCosmo: AI Skincare Recommender",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Reverted Card Styles */
    .product-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .buy-button {
        display: inline-block;
        background-color: #ff4b4b;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        width: 100%;
        margin-top: 10px;
    }
    .buy-button:hover {
        background-color: #ff3333;
        color: white;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    h1, h2, h3 {
        color: #333;
    }
    .product-card h3 {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("skindataall.csv")
        # Preprocessing
        df = df.dropna(subset=["Product", "Price", "Rating", "Skin_Type", "Skin_Tone"])
        df["Price"] = df["Price"].astype(float)
        df["Rating"] = df["Rating"].astype(float)
        df["Good_Stuff"] = df["Good_Stuff"].astype(int)
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file 'skindataall.csv' not found. Please ensure it is in the project directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Key Ingredients Dictionary for Highlights ---
KEY_INGREDIENTS = {
    "Vitamin C": "Brightening & Anti-aging",
    "Hyaluronic Acid": "Hydration",
    "Salicylic Acid": "Acne fighting",
    "Retinol": "Anti-aging",
    "Niacinamide": "Pore refining & Soothing",
    "Glycolic Acid": "Exfoliation",
    "Aloe Vera": "Soothing",
    "Ceramides": "Barrier repair",
    "Squalane": "Moisturizing",
    "Tea Tree Oil": "Acne fighting"
}

def highlight_ingredients(ingredients_text):
    found = []
    if pd.isna(ingredients_text):
        return found
    text_lower = ingredients_text.lower()
    for ingredient, benefit in KEY_INGREDIENTS.items():
        if ingredient.lower() in text_lower:
            found.append(f"**{ingredient}** ({benefit})")
    return found

# --- Sidebar Filters ---
with st.sidebar:
    st.title("💄 User Preferences")
    st.markdown("Customize your skincare search.")
    
    with st.expander("👤 Personal Details", expanded=True):
        skin_type = st.selectbox("🧴 Skin Type:", sorted(df["Skin_Type"].dropna().unique()))
        skin_tone = st.selectbox("🌈 Skin Tone:", sorted(df["Skin_Tone"].dropna().unique()))
        
    with st.expander("🛍️ Product Preferences", expanded=True):
        product_type = st.selectbox("🧼 Category:", sorted(df["Category"].dropna().unique()))
        max_price = st.slider("💰 Max Budget ($):", min_value=1, max_value=int(df["Price"].max()), value=50)
        
    with st.expander("🚫 Allergies & Exclusions"):
        allergy_input = st.text_input("Enter ingredients to avoid (comma-separated):", placeholder="e.g., alcohol, fragrance")
        allergies = [a.strip().lower() for a in allergy_input.split(",") if a.strip()]

    st.markdown("---")
    st.info("💡 **Tip:** Adjust filters to see more results.")

# --- Main Layout ---
st.title("💄 SmartCosmo: Intelligent Skincare Recommender")
st.markdown("### 🤖 AI-Powered Personalized Skincare Recommendations")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["🌟 Recommendations", "📊 Market Analytics", "ℹ️ About the Project"])

# --- Tab 1: Recommendations ---
with tab1:
    # Filtering logic
    filtered_df = df.copy()
    
    # Allergy Filter
    if allergies:
        filtered_df = filtered_df[~filtered_df["Ingredients_Cleaned"].astype(str).apply(
            lambda x: any(a in x.lower() for a in allergies)
        )]

    # Preference Filter
    filtered_df = filtered_df[
        (filtered_df["Category"] == product_type) &
        (filtered_df["Skin_Type"] == skin_type) &
        (filtered_df["Skin_Tone"] == skin_tone) &
        (filtered_df["Price"] <= max_price) &
        (filtered_df["Good_Stuff"] == 1) # Only positive sentiment
    ]

    if filtered_df.empty:
        st.warning(f"🤔 No exact matches found for **{skin_type}** skin, **{skin_tone}** tone, under **${max_price}** in **{product_type}**.")
        st.markdown("Try increasing your budget or changing the filters.")
    else:
        # --- Algorithm: TF-IDF + Cosine Similarity ---
        # Using Review + Ingredients for similarity
        tfidf = TfidfVectorizer(stop_words='english')
        text_features = filtered_df["Review_Cleaned"].fillna('') + " " + filtered_df["Ingredients_Cleaned"].fillna('')
        
        if len(filtered_df) > 1:
            tfidf_matrix = tfidf.fit_transform(text_features)
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # Similarity within the filtered subset
            filtered_df["Text_Similarity"] = cosine_sim.mean(axis=1) # Average similarity to other "good" products in this niche
        else:
            filtered_df["Text_Similarity"] = 1.0 # Default if only one result

        # Scoring System
        max_rating = filtered_df["Rating"].max()
        filtered_df["Price_Score"] = 1 - (filtered_df["Price"] / (max_price + 1)) # Lower price is better (relative to budget)
        filtered_df["Rating_Score"] = filtered_df["Rating"] / max_rating if max_rating > 0 else 0
        
        # Weighted Final Score
        filtered_df["Final_Score"] = (
            0.4 * filtered_df["Rating_Score"] +
            0.4 * filtered_df["Price_Score"] +
            0.2 * filtered_df["Text_Similarity"]
        )

        # Sort and Deduplicate
        filtered_df = filtered_df.sort_values("Final_Score", ascending=False)
        top_picks = filtered_df.drop_duplicates(subset=["Product"]).head(5)

        st.success(f"✅ Found {len(top_picks)} top recommendations for you.")
        
        # Display in a grid
        cols = st.columns(2)
        
        for i, (index, row) in enumerate(top_picks.iterrows()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="product-card">
                    <h3 style="margin-bottom:5px;">{i+1}. {row['Product']}</h3>
                    <p style="color:gray; font-size:0.9em; margin-bottom:15px;">By <b>{row['Brand']}</b></p>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("💵 Price", f"${row['Price']:.2f}")
                c2.metric("⭐ Rating", f"{row['Rating']}/5.0")
                c3.metric("📈 Match", f"{row['Final_Score']*100:.0f}%")
                
                # Ingredient Highlights
                highlights = highlight_ingredients(row['Ingredients'])
                if highlights:
                    st.markdown("<br><b>🧪 Key Ingredients:</b>", unsafe_allow_html=True)
                    for h in highlights[:3]: # Show max 3 highlights
                         st.markdown(f"- {h}")
                
                st.markdown("---")
                
                with st.expander("📄 Detail View"):
                    st.info(f"💬 \"{str(row['Review'])[:150]}...\"")
                    st.text_area("Ingredients:", row['Ingredients'], height=80, key=f"ing_{index}")

                st.markdown(f"""
                    <a href="{row['Product_Url']}" target="_blank" class="buy-button">
                        🛒 Buy Now
                    </a>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 2: Analytics ---
with tab2:
    st.header("📊 Market Insights Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["Good_Stuff"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        sentiment_counts["Sentiment"] = sentiment_counts["Sentiment"].map({1: "Positive", 0: "Negative"})
        
        fig_pie = px.pie(sentiment_counts, values="Count", names="Sentiment", 
                         color="Sentiment", color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B"},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Price vs. Rating by Skin Type")
        fig_scatter = px.scatter(df, x="Price", y="Rating", color="Skin_Type",
                                 hover_data=["Product", "Brand"],
                                 opacity=0.6, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("🏆 Average Rating by Product Category")
    avg_rating = df.groupby("Category")["Rating"].mean().reset_index().sort_values("Rating", ascending=False)
    fig_bar = px.bar(avg_rating, x="Category", y="Rating", color="Rating", 
                     color_continuous_scale="Viridis", text_auto=".2f")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 3: About ---
with tab3:
    st.header("ℹ️ About This Project")
    st.markdown("""
    ### **Project Information**
    This project is a **Natural Language Processing (NLP) based Recommendation System** designed to suggest the best skincare products tailored to a user's specific needs (Skin Type, Tone, Budget).

    ### **How It Works (The Algorithm)**
    1.  **Data Preprocessing**:
        -   Cleans text data (reviews, ingredients).
        -   Filters products containing allergens.
        -   Filters for only "Positive" sentiment products based on the `Good_Stuff` classification.
    
    2.  **Content-Based Filtering**:
        -   Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize product descriptions, reviews, and ingredient lists.
        -   Calculates **Cosine Similarity** to find products that are textually similar to the "ideal" product profile in the filtered subset.
    
    3.  **Hybrid Scoring System**:
        -   A final `Match Score` is computed using a weighted formula:
        $$ Score = 0.4 \\times NormalizedRating + 0.4 \\times PriceScore + 0.2 \\times TextSimilarity $$
        -   This ensures recommendations are not just popular (high rating), but also affordable and contextually relevant.
    
    ### **Tech Stack**
    -   **Frontend**: Streamlit (Python)
    -   **Data Processing**: Pandas
    -   **Machine Learning**: Scikit-Learn (TF-IDF, Cosine Similarity)
    -   **Visualization**: Plotly Express
    """)