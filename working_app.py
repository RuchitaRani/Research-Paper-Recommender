import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Research Paper Classification & Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_processed_data():
    """Load the preprocessed ArXiv data"""
    try:
        # Load processed papers
        papers_df = pd.read_pickle("models/processed_papers.pkl")
        
        # Load multilabel binarizer for categories
        with open("models/vocab.pkl", "rb") as f:
            mlb = pickle.load(f)
        
        return papers_df, mlb
    except FileNotFoundError:
        return None, None

def simple_classify(text, mlb):
    """Simple keyword-based classification"""
    if mlb is None:
        return []
    
    text_lower = text.lower()
    
    # Common keywords for major categories
    category_keywords = {
        'cs.CV': ['image', 'vision', 'visual', 'computer vision', 'detection', 'recognition', 'segmentation', 'convolutional'],
        'cs.LG': ['machine learning', 'deep learning', 'neural network', 'learning', 'training', 'model', 'algorithm'],
        'cs.AI': ['artificial intelligence', 'ai', 'intelligent', 'reasoning', 'decision', 'autonomous'],
        'cs.CL': ['natural language', 'nlp', 'text', 'language', 'linguistic', 'translation', 'transformer'],
        'cs.RO': ['robot', 'robotics', 'navigation', 'control', 'autonomous', 'mobile robot'],
        'stat.ML': ['statistical', 'statistics', 'forecasting', 'time series', 'analysis', 'performance'],
        'cs.CR': ['security', 'privacy', 'cryptography', 'attack', 'defense', 'adversarial'],
        'cs.DC': ['distributed', 'parallel', 'edge', 'computing', 'cloud', 'federated'],
        'cs.NE': ['neural', 'evolution', 'genetic', 'evolutionary', 'optimization'],
        'eess.IV': ['signal', 'processing', 'medical', 'imaging', 'biomedical']
    }
    
    predictions = []
    for category in mlb.classes_[:50]:  # Check top 50 categories
        if category in category_keywords:
            keywords = category_keywords[category]
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                normalized_score = min(score / len(keywords), 1.0)
                predictions.append((category, normalized_score))
    
    # Sort by score
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def simple_recommend(query, papers_df, top_k=5):
    """Simple TF-IDF based recommendation"""
    if papers_df is None:
        return []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Combine titles and abstracts for better matching
    documents = papers_df['titles'] + ' ' + papers_df['abstracts'].fillna('')
    
    # Create corpus with query + papers
    corpus = [query] + documents.tolist()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate similarities (query is first item)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get top recommendations
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    recommendations = []
    for i, idx in enumerate(top_indices):
        recommendations.append({
            'title': papers_df.iloc[idx]['titles'],
            'similarity': similarities[idx],
            'rank': i + 1
        })
    
    return recommendations

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("📚 Research Paper Classification & Recommendation System")
    st.markdown("""
    This system provides two main functionalities:
    1. **Subject Area Classification**: Predict academic categories for research paper abstracts
    2. **Paper Recommendation**: Find similar research papers based on paper titles
    """)
    
    # Try to load data
    papers_df, mlb = load_processed_data()
    
    if papers_df is None:
        st.warning("⚠️ Preprocessed data not found. Please run the preprocessing first:")
        st.code("python data_preprocessing.py")
        st.info("💡 The system will create sample data for demonstration.")
        
        # Create sample data for demo
        sample_data = {
            'titles': [
                'Deep Learning for Computer Vision Applications',
                'Natural Language Processing with Transformers',
                'Reinforcement Learning for Robotics',
                'Machine Learning in Healthcare',
                'Computer Vision for Autonomous Vehicles'
            ],
            'abstracts': [
                'This paper explores deep learning techniques for computer vision tasks including image classification and object detection.',
                'We present a comprehensive study of transformer models for natural language processing applications.',
                'This work investigates reinforcement learning methods for robotic control and navigation.',
                'We examine machine learning applications in medical diagnosis and treatment planning.',
                'This research focuses on computer vision systems for autonomous vehicle navigation.'
            ],
            'filtered_terms': [
                ['cs.CV', 'cs.LG'],
                ['cs.CL', 'cs.AI'],
                ['cs.RO', 'cs.AI'],
                ['cs.LG', 'cs.AI'],
                ['cs.CV', 'cs.RO']
            ]
        }
        papers_df = pd.DataFrame(sample_data)
        
        # Create simple mlb for demo
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit(papers_df['filtered_terms'])
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a function:",
        ["🏠 Home", "🔍 Classification", "💡 Recommendations", "📊 System Info"]
    )
    
    if page == "🏠 Home":
        show_home_page(papers_df, mlb)
    elif page == "🔍 Classification":
        show_classification_page(mlb)
    elif page == "💡 Recommendations":
        show_recommendation_page(papers_df)
    elif page == "📊 System Info":
        show_system_info(papers_df, mlb)

def show_home_page(papers_df, mlb):
    """Show home page"""
    
    st.header("Welcome to the Research Paper System! 🚀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Classification System")
        st.write("""
        Our classification system predicts academic subject areas:
        
        - **Keyword-based matching** with domain expertise
        - **Multi-label classification** for overlapping categories
        - **Real-time predictions** with confidence scores
        - **ArXiv taxonomy** (cs.AI, cs.LG, cs.CV, etc.)
        """)
    
    with col2:
        st.subheader("💡 Recommendation System")
        st.write("""
        Our recommendation system finds similar papers:
        
        - **TF-IDF vectorization** of paper content
        - **Cosine similarity** for ranking
        - **Real-time recommendations** with similarity scores
        - **Semantic matching** beyond keyword overlap
        """)
    
    # System stats
    st.header("📊 System Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Papers in Database", f"{len(papers_df):,}")
    
    with col2:
        if mlb:
            st.metric("Categories Available", len(mlb.classes_))
        else:
            st.metric("Categories Available", "N/A")
    
    with col3:
        st.metric("Classification Method", "Keyword-based")
    
    with col4:
        st.metric("Recommendation Method", "TF-IDF Similarity")
    
    # Sample papers
    st.header("📄 Sample Papers")
    st.dataframe(papers_df[['titles']].head(10), use_container_width=True)

def show_classification_page(mlb):
    """Show classification page"""
    
    st.header("🔍 Research Paper Classification")
    
    # Example abstracts
    example_abstracts = {
        "Deep Learning Example": """
        This paper presents a comprehensive study on the application of deep neural networks 
        for image classification tasks. We propose a novel convolutional architecture that 
        combines residual connections with attention mechanisms to achieve state-of-the-art 
        performance on benchmark datasets. Our experimental results demonstrate significant 
        improvements in both accuracy and computational efficiency compared to existing methods.
        """,
        
        "NLP Example": """
        Natural language processing has been revolutionized by transformer-based models. 
        In this work, we introduce a new approach for sentiment analysis that leverages 
        pre-trained language models and fine-tuning techniques. We evaluate our method 
        on multiple social media datasets and show substantial improvements in classification 
        accuracy while maintaining computational efficiency.
        """,
        
        "Robotics Example": """
        Autonomous navigation in complex environments remains a challenging problem in robotics. 
        This paper presents a reinforcement learning approach for mobile robot navigation 
        that combines deep Q-learning with computer vision. Our method enables robots to 
        learn navigation policies through interaction with the environment, avoiding 
        obstacles and reaching target destinations efficiently.
        """
    }
    
    # Example selector
    selected_example = st.selectbox(
        "Choose an example abstract (optional):",
        ["None"] + list(example_abstracts.keys())
    )
    
    # Text input
    if selected_example != "None":
        default_text = example_abstracts[selected_example].strip()
    else:
        default_text = ""
    
    abstract = st.text_area(
        "Enter paper abstract:",
        value=default_text,
        height=200,
        placeholder="Paste your research paper abstract here..."
    )
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.2, 0.05)
    
    with col2:
        max_categories = st.number_input("Max Categories", 1, 20, 10)
    
    # Classify button
    if st.button("🔍 Classify Paper", type="primary"):
        if abstract.strip():
            with st.spinner("Analyzing abstract..."):
                predictions = simple_classify(abstract, mlb)
                
                # Filter by threshold
                filtered_predictions = [(cat, score) for cat, score in predictions if score >= threshold]
                
                if filtered_predictions:
                    st.success(f"Found {len(filtered_predictions)} relevant categories:")
                    
                    for i, (category, score) in enumerate(filtered_predictions[:max_categories]):
                        col1, col2, col3 = st.columns([3, 1, 2])
                        
                        with col1:
                            st.write(f"**{category}**")
                        with col2:
                            st.write(f"{score:.3f}")
                        with col3:
                            st.progress(min(score, 1.0))
                    
                    # Details
                    with st.expander("🔬 Prediction Details"):
                        df_predictions = pd.DataFrame(filtered_predictions[:max_categories], 
                                                    columns=['Category', 'Score'])
                        st.dataframe(df_predictions, use_container_width=True)
                else:
                    st.warning(f"No categories found above threshold {threshold}.")
        else:
            st.warning("Please enter an abstract to classify.")

def show_recommendation_page(papers_df):
    """Show recommendation page"""
    
    st.header("💡 Research Paper Recommendations")
    
    # Example queries
    example_queries = [
        "Deep learning for computer vision",
        "Natural language processing transformers",
        "Machine learning healthcare applications",
        "Reinforcement learning robotics"
    ]
    
    # Query input
    query = st.text_input(
        "Enter paper title or research topic:",
        placeholder="e.g., 'Deep learning for image recognition'"
    )
    
    # Add example buttons
    st.write("**Quick examples:**")
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        if cols[i].button(f"📝 {example.split()[0]} {example.split()[1]}", key=f"example_{i}"):
            query = example
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider("Number of Recommendations", 1, 10, 5)
    
    with col2:
        min_similarity = st.slider("Minimum Similarity", 0.0, 1.0, 0.0, 0.05)
    
    # Recommend button
    if st.button("💡 Find Similar Papers", type="primary") or query:
        if query.strip():
            with st.spinner("Finding similar papers..."):
                recommendations = simple_recommend(query, papers_df, top_k=top_k)
                
                # Filter by similarity threshold
                filtered_recs = [rec for rec in recommendations if rec['similarity'] >= min_similarity]
                
                if filtered_recs:
                    st.success(f"Found {len(filtered_recs)} similar papers:")
                    
                    for rec in filtered_recs:
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.write(f"**{rec['rank']}. {rec['title']}**")
                        
                        with col2:
                            similarity = rec['similarity']
                            if similarity >= 0.3:
                                st.success(f"🎯 {similarity:.3f}")
                            elif similarity >= 0.1:
                                st.info(f"✅ {similarity:.3f}")
                            else:
                                st.write(f"📄 {similarity:.3f}")
                        
                        st.progress(min(similarity * 3, 1.0))  # Scale for visibility
                        st.divider()
                    
                    # Details
                    with st.expander("🔬 Recommendation Details"):
                        df_recs = pd.DataFrame([
                            {
                                'Rank': rec['rank'],
                                'Title': rec['title'],
                                'Similarity': rec['similarity']
                            }
                            for rec in recommendations
                        ])
                        st.dataframe(df_recs, use_container_width=True)
                else:
                    st.warning(f"No papers found above similarity threshold {min_similarity}.")
        else:
            st.info("Enter a query above to get recommendations.")

def show_system_info(papers_df, mlb):
    """Show system information"""
    
    st.header("📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Classification System")
        if mlb:
            st.write(f"**Categories**: {len(mlb.classes_)}")
            st.write("**Method**: Keyword-based matching")
            st.write("**Real-time**: ✅ No training required")
            
            with st.expander("Top Categories"):
                for cat in mlb.classes_[:20]:
                    st.write(f"• {cat}")
        else:
            st.write("**Status**: No model loaded")
    
    with col2:
        st.subheader("💡 Recommendation System")
        st.write(f"**Papers**: {len(papers_df):,}")
        st.write("**Method**: TF-IDF + Cosine Similarity")
        st.write("**Real-time**: ✅ Instant recommendations")
        
        with st.expander("Sample Papers"):
            st.dataframe(papers_df[['titles']].head(10), use_container_width=True)
    
    # File status
    st.subheader("📁 System Files")
    
    files_to_check = {
        "Processed Papers": "models/processed_papers.pkl",
        "Category Vocabulary": "models/vocab.pkl",
        "Data Splits": "models/data_splits.pkl",
        "Classification Model": "models/model.h5"
    }
    
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        if exists:
            try:
                size = f"{os.path.getsize(path) / 1024 / 1024:.1f} MB"
            except:
                size = "Directory"
        else:
            size = "Missing"
        
        st.write(f"**{name}**: {'✅' if exists else '❌'} {size}")
    
    # Instructions
    st.subheader("🚀 Getting Started")
    st.write("""
    1. **Data**: Run `python data_preprocessing.py` to process ArXiv data
    2. **Models**: Run `python classification_model.py` to train ML models
    3. **Recommendations**: Run `python recommendation_system.py` for embeddings
    4. **Web App**: This interface works with or without trained models
    """)

if __name__ == "__main__":
    main()