# 💄 SmartCosmo: AI Skincare Recommendation System

## 📌 Project Overview
SmartCosmo is an AI-powered skincare recommendation system that suggests personalized skincare products based on user preferences such as skin type, skin tone, product category, budget, and ingredient allergies.

The system uses **Natural Language Processing (NLP)** techniques to analyze product reviews and ingredient lists, enabling intelligent recommendations using **TF-IDF vectorization and cosine similarity**.

The application is built as an interactive **web app using Streamlit** with additional analytics dashboards.

---

## 🚀 Features

- Personalized skincare product recommendations
- Ingredient allergy filtering
- AI-based similarity matching
- Budget-based filtering
- Skin type and skin tone customization
- Interactive product cards with ratings and price
- Market analytics dashboard

---

## ⚙️ How the System Works

### 1. Data Preprocessing
- Clean product reviews and ingredient lists
- Remove missing values
- Convert price and ratings to numerical format

### 2. Content-Based Recommendation
- TF-IDF vectorization of product reviews and ingredients
- Cosine similarity calculation between products

### 3. Hybrid Scoring System
Products are ranked using a weighted scoring formula:

Score = 0.4 × Rating Score  
+ 0.4 × Price Score  
+ 0.2 × Text Similarity

This ensures recommendations are:
- Highly rated
- Affordable
- Textually similar to the ideal skincare profile

---

## 📊 Analytics Dashboard

The system also includes visual analytics:

- Sentiment distribution of skincare products
- Price vs rating scatter plot
- Average rating by product category

Visualizations are created using **Plotly**.

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Plotly
- Natural Language Processing (NLP)

---

## 📂 Project Structure
SmartCosmo/
│
├── SkincareRecommendation.py
├── skindataall.csv
├── README.md
└── requirements.txt

