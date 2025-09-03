import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

def create_sentence_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """Create sentence embeddings using SentenceTransformer"""
    
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    
    # Convert to CPU numpy array for storage
    embeddings_np = embeddings.cpu().numpy()
    
    print(f"Embeddings shape: {embeddings_np.shape}")
    return model, embeddings_np

def save_recommendation_artifacts(model, embeddings, sentences):
    """Save recommendation system artifacts"""
    
    print("Saving recommendation artifacts...")
    os.makedirs("models", exist_ok=True)
    
    # Save embeddings
    with open("models/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    # Save sentences
    with open("models/sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)
    
    # Save model
    model.save("models/sentence_transformer")
    
    print("Recommendation artifacts saved successfully!")

def load_recommendation_artifacts():
    """Load recommendation system artifacts"""
    
    # Load embeddings
    with open("models/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    
    # Load sentences
    with open("models/sentences.pkl", "rb") as f:
        sentences = pickle.load(f)
    
    # Load model
    model = SentenceTransformer("models/sentence_transformer")
    
    return model, embeddings, sentences

def find_similar_papers(query_title, model=None, embeddings=None, sentences=None, top_k=5):
    """Find similar papers based on title similarity"""
    
    if model is None or embeddings is None or sentences is None:
        model, embeddings, sentences = load_recommendation_artifacts()
    
    # Generate embedding for query
    query_embedding = model.encode([query_title], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding_np, embeddings)[0]
    
    # Get top-k most similar papers
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'title': sentences[idx],
            'similarity': float(similarities[idx]),
            'rank': len(results) + 1
        })
    
    return results

def build_recommendation_system():
    """Build the recommendation system"""
    
    print("Building recommendation system...")
    
    # Load processed papers
    try:
        papers_df = pd.read_pickle("models/processed_papers.pkl")
        print(f"Loaded {len(papers_df)} papers for recommendation system")
    except FileNotFoundError:
        print("Processed papers not found. Creating sample data...")
        
        # Create sample titles for demonstration
        sample_titles = [
            "Deep Learning for Time Series Forecasting with Long Short-Term Memory Networks",
            "Artificial Intelligence in Healthcare: A Comprehensive Survey of Machine Learning Applications",
            "Computer Vision Techniques for Autonomous Vehicle Navigation and Object Detection",
            "Neural Evolution Strategies for Optimizing Deep Neural Network Architectures",
            "Statistical Machine Learning Methods for High-Dimensional Data Analysis",
            "Explainable AI: Methods and Applications in Critical Decision Making Systems",
            "Advanced Image Processing Techniques for Medical Imaging and Diagnosis",
            "Robotic Process Automation Using AI-Driven Decision Making Algorithms",
            "Natural Language Processing for Sentiment Analysis in Social Media Data",
            "Machine Learning Approaches for Predictive Analytics in Business Intelligence",
            "Database Optimization Using AI-Based Query Processing and Indexing Strategies",
            "Information Retrieval Systems Enhanced with Natural Language Understanding",
            "Human-Computer Interaction Design Principles for AI-Assisted User Interfaces",
            "Control Systems Engineering with Machine Learning for Autonomous Systems",
            "Cybersecurity Applications of Artificial Intelligence in Threat Detection",
            "Network Intelligence: AI Applications in Telecommunications and IoT",
            "Data Structures and Algorithms for Large-Scale Statistical Computing",
            "Game Theory Applications in Multi-Agent Artificial Intelligence Systems",
            "Distributed Computing Frameworks for Scalable Machine Learning",
            "Software Engineering Practices for AI System Development and Deployment",
            "Reinforcement Learning for Robot Control and Navigation",
            "Transformer Models for Machine Translation and Language Understanding",
            "Convolutional Neural Networks for Medical Image Segmentation",
            "Federated Learning: Privacy-Preserving Machine Learning Across Distributed Data",
            "Graph Neural Networks for Social Network Analysis and Recommendation Systems",
            "Attention Mechanisms in Deep Learning for Sequence-to-Sequence Tasks",
            "Generative Adversarial Networks for Synthetic Data Generation",
            "Transfer Learning Techniques for Few-Shot Learning Applications",
            "Multimodal Learning: Integrating Text, Image, and Audio Data",
            "Quantum Machine Learning: Algorithms and Applications"
        ]
        
        papers_df = pd.DataFrame({'titles': sample_titles})
        papers_df.to_pickle("models/processed_papers.pkl")
        print(f"Created sample dataset with {len(sample_titles)} papers")
    
    # Extract titles
    titles = papers_df['titles'].tolist()
    
    # Create embeddings
    model, embeddings = create_sentence_embeddings(titles)
    
    # Save artifacts
    save_recommendation_artifacts(model, embeddings, titles)
    
    print("Recommendation system built successfully!")
    return model, embeddings, titles

def test_recommendation_system():
    """Test the recommendation system"""
    
    print("\nTesting recommendation system...")
    
    # Test queries
    test_queries = [
        "Deep learning for computer vision",
        "Natural language processing with transformers",
        "Machine learning for healthcare applications",
        "Reinforcement learning in robotics"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        similar_papers = find_similar_papers(query, top_k=3)
        
        for paper in similar_papers:
            print(f"  {paper['rank']}. {paper['title']}")
            print(f"     Similarity: {paper['similarity']:.3f}")

def main():
    """Main recommendation system pipeline"""
    
    # Check if artifacts exist
    if os.path.exists("models/embeddings.pkl"):
        print("Recommendation artifacts found. Loading...")
        try:
            model, embeddings, sentences = load_recommendation_artifacts()
            print("Recommendation system loaded successfully!")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Rebuilding recommendation system...")
            model, embeddings, sentences = build_recommendation_system()
    else:
        print("Building recommendation system from scratch...")
        model, embeddings, sentences = build_recommendation_system()
    
    # Test the system
    test_recommendation_system()

if __name__ == "__main__":
    main()