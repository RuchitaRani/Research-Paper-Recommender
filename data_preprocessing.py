import pandas as pd
import numpy as np
import ast
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def download_arxiv_data():
    """Use local ArXiv dataset from data/archive folder"""
    try:
        # Check for the dataset in the local archive folder
        local_path = "data/archive/arxiv_data_210930-054931.csv"
        if os.path.exists(local_path):
            print(f"Using local dataset: {local_path}")
            return local_path
        
        # Try alternative name
        alt_path = "data/archive/arxiv_data.csv"
        if os.path.exists(alt_path):
            print(f"Using local dataset: {alt_path}")
            return alt_path
            
        print("Local dataset not found in data/archive/")
        # If local file not found, create a sample dataset for demonstration
        return create_sample_data()
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        # If loading fails, create a sample dataset for demonstration
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes"""
    print("Creating sample ArXiv dataset...")
    
    sample_data = {
        'terms': [
            "['cs.LG', 'stat.ML']",
            "['cs.AI', 'cs.LG']",
            "['cs.CV', 'cs.LG']",
            "['cs.NE', 'cs.AI']",
            "['stat.ML', 'cs.LG']",
            "['cs.AI']",
            "['cs.CV']",
            "['cs.RO', 'cs.AI']",
            "['cs.CL', 'cs.AI']",
            "['cs.LG']",
            "['cs.DB', 'cs.AI']",
            "['cs.IR', 'cs.CL']",
            "['cs.HC', 'cs.AI']",
            "['cs.SY', 'cs.LG']",
            "['cs.CR', 'cs.AI']",
            "['cs.NI', 'cs.LG']",
            "['cs.DS', 'stat.ML']",
            "['cs.GT', 'cs.AI']",
            "['cs.DC', 'cs.LG']",
            "['cs.SE', 'cs.AI']"
        ],
        'titles': [
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
            "Software Engineering Practices for AI System Development and Deployment"
        ],
        'abstracts': [
            "This paper presents a comprehensive study on the application of Long Short-Term Memory (LSTM) networks for time series forecasting. We investigate the effectiveness of deep learning approaches in capturing temporal dependencies and propose novel architectures for improving prediction accuracy. Our experimental results demonstrate significant improvements over traditional statistical methods across multiple benchmark datasets.",
            "Healthcare systems are increasingly adopting artificial intelligence technologies to improve patient outcomes and operational efficiency. This survey examines the current state of machine learning applications in medical diagnosis, treatment planning, drug discovery, and personalized medicine. We analyze the challenges and opportunities in implementing AI solutions in clinical environments.",
            "Autonomous vehicles rely heavily on computer vision systems for safe navigation and decision-making. This research explores state-of-the-art techniques in object detection, semantic segmentation, and depth estimation for automotive applications. We present a comprehensive evaluation of different neural network architectures and their performance in real-world driving scenarios.",
            "Neural architecture search has emerged as a powerful technique for automating the design of deep neural networks. This work introduces novel evolutionary strategies for optimizing network topologies, exploring the trade-offs between accuracy and computational efficiency. Our approach demonstrates superior performance on image classification and natural language processing tasks.",
            "High-dimensional data analysis presents unique challenges in statistical machine learning. This paper investigates dimensionality reduction techniques, feature selection methods, and regularization strategies for handling datasets with thousands of variables. We provide theoretical foundations and practical guidelines for practitioners working with complex data structures.",
            "The increasing deployment of AI systems in critical applications necessitates transparent and interpretable models. This research examines various explainable AI techniques, including attention mechanisms, gradient-based methods, and model-agnostic approaches. We evaluate their effectiveness in providing meaningful insights into AI decision-making processes.",
            "Medical imaging has been revolutionized by advanced image processing techniques powered by deep learning. This study focuses on applications in radiology, pathology, and medical diagnostics, examining how AI-enhanced imaging can improve diagnostic accuracy and reduce clinical workload. We present case studies from multiple medical specialties.",
            "Robotic process automation is being enhanced by artificial intelligence to handle complex decision-making tasks. This paper explores the integration of machine learning algorithms with robotic systems for intelligent automation in manufacturing, logistics, and service industries. We demonstrate the benefits of AI-driven robotics in improving efficiency and adaptability.",
            "Social media platforms generate vast amounts of textual data that require sophisticated natural language processing techniques for analysis. This research investigates sentiment analysis methods, including transformer-based models and ensemble approaches. We evaluate performance across different social media platforms and discuss challenges in handling multilingual and multimodal content.",
            "Predictive analytics in business intelligence leverages machine learning to forecast trends and support strategic decision-making. This work examines various predictive modeling techniques, their applications in different business domains, and the challenges of deploying machine learning systems in enterprise environments. We provide practical recommendations for successful implementation.",
            "Database systems can benefit significantly from AI-driven optimization techniques. This research explores machine learning approaches for query optimization, indexing strategies, and database tuning. We investigate how artificial intelligence can improve database performance and reduce administrative overhead in large-scale data management systems.",
            "Information retrieval systems are being enhanced with natural language understanding capabilities to improve search accuracy and user satisfaction. This paper examines the integration of semantic search, question-answering systems, and conversational interfaces. We evaluate different approaches and their effectiveness in various information retrieval scenarios.",
            "The design of user interfaces for AI-assisted systems requires careful consideration of human-computer interaction principles. This research explores design patterns, usability guidelines, and evaluation methods for AI-enhanced interfaces. We present case studies and best practices for creating intuitive and effective AI-human collaboration systems.",
            "Control systems in autonomous applications are increasingly incorporating machine learning for adaptive and intelligent behavior. This work investigates the application of reinforcement learning and adaptive control techniques in robotics, autonomous vehicles, and industrial automation. We analyze stability, safety, and performance considerations.",
            "Cybersecurity threats are evolving rapidly, requiring advanced artificial intelligence techniques for effective detection and prevention. This research examines machine learning approaches for threat intelligence, anomaly detection, and automated incident response. We evaluate the effectiveness of different AI methods in real-world cybersecurity scenarios.",
            "Telecommunications and IoT networks can leverage artificial intelligence for improved performance and intelligent resource management. This paper explores applications of machine learning in network optimization, traffic prediction, and quality of service management. We present experimental results from large-scale network deployments.",
            "Large-scale statistical computing requires efficient data structures and algorithms to handle massive datasets. This research investigates optimization techniques for statistical computations, parallel processing strategies, and memory-efficient algorithms. We demonstrate significant performance improvements in various statistical analysis tasks.",
            "Multi-agent systems in artificial intelligence can benefit from game theory principles for coordination and decision-making. This work explores the application of game-theoretic concepts in distributed AI systems, auction mechanisms, and resource allocation problems. We analyze equilibrium properties and convergence guarantees.",
            "Scalable machine learning requires distributed computing frameworks that can handle large datasets and complex models. This research examines different distributed learning approaches, including federated learning and parameter servers. We evaluate their performance, communication efficiency, and fault tolerance in large-scale deployments.",
            "The development and deployment of AI systems require specialized software engineering practices and methodologies. This paper investigates best practices for AI system architecture, testing strategies, and continuous integration in machine learning pipelines. We provide guidelines for managing the unique challenges of AI software development."
        ]
    }
    
    # Save sample data
    df = pd.DataFrame(sample_data)
    sample_path = "data/sample_arxiv_data.csv"
    df.to_csv(sample_path, index=False)
    print(f"Sample dataset created at: {sample_path}")
    return sample_path

def preprocess_data(csv_path):
    """Preprocess the ArXiv dataset"""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers")
    
    # Remove duplicates based on titles
    df = df.drop_duplicates(subset=['titles'], keep='first')
    print(f"After removing duplicates: {len(df)} papers")
    
    # Convert string representations of lists to actual lists
    def safe_eval(x):
        try:
            if isinstance(x, str):
                # Clean the string and convert to list
                x = x.strip("[]").replace("'", "").replace('"', '')
                return [term.strip() for term in x.split(',') if term.strip()]
            elif isinstance(x, list):
                return x
            else:
                return []
        except:
            return []
    
    df['terms'] = df['terms'].apply(safe_eval)
    
    # Filter out papers with no terms
    df = df[df['terms'].apply(len) > 0]
    print(f"After filtering papers with no terms: {len(df)} papers")
    
    # Count term frequencies
    all_terms = []
    for terms in df['terms']:
        all_terms.extend(terms)
    
    term_counts = pd.Series(all_terms).value_counts()
    print(f"Total unique terms: {len(term_counts)}")
    
    # Keep only terms that appear at least 2 times
    valid_terms = set(term_counts[term_counts >= 2].index)
    print(f"Terms appearing >= 2 times: {len(valid_terms)}")
    
    # Filter papers to only include valid terms
    df['filtered_terms'] = df['terms'].apply(
        lambda x: [term for term in x if term in valid_terms]
    )
    
    # Remove papers with no valid terms after filtering
    df = df[df['filtered_terms'].apply(len) > 0]
    print(f"Final dataset size: {len(df)} papers")
    
    # Create multilabel binarizer
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(df['filtered_terms'])
    
    print(f"Number of unique categories: {len(mlb.classes_)}")
    print(f"Category distribution (top 10):")
    category_sums = y_encoded.sum(axis=0)
    top_categories = sorted(zip(mlb.classes_, category_sums), 
                          key=lambda x: x[1], reverse=True)[:10]
    for cat, count in top_categories:
        print(f"  {cat}: {count}")
    
    return df, mlb, y_encoded

def create_text_vectorizer(texts, max_features=10000):
    """Create and fit TF-IDF vectorizer"""
    print("Creating text vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        stop_words='english',
        lowercase=True,
        min_df=2,  # minimum document frequency
        max_df=0.95  # maximum document frequency
    )
    
    X_vectorized = vectorizer.fit_transform(texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix shape: {X_vectorized.shape}")
    
    return vectorizer, X_vectorized

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/validation/test sets"""
    print("Splitting data...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def save_preprocessing_artifacts(vectorizer, mlb, data_splits, label_splits):
    """Save preprocessing artifacts for later use"""
    print("Saving preprocessing artifacts...")
    
    os.makedirs("models", exist_ok=True)
    
    # Save vectorizer
    with open("models/text_vectorizer_config.pkl", "wb") as f:
        pickle.dump({
            'vocabulary_': vectorizer.vocabulary_,
            'idf_': vectorizer.idf_,
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
            'stop_words': vectorizer.stop_words,
            'lowercase': vectorizer.lowercase,
            'min_df': vectorizer.min_df,
            'max_df': vectorizer.max_df
        }, f)
    
    # Save multilabel binarizer
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(mlb, f)
    
    # Save data splits
    X_train, X_val, X_test = data_splits
    y_train, y_val, y_test = label_splits
    
    with open("models/data_splits.pkl", "wb") as f:
        pickle.dump({
            'X_train': X_train,
            'X_val': X_val, 
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }, f)
    
    print("Preprocessing artifacts saved successfully!")

def main():
    """Main preprocessing pipeline"""
    # Download data
    csv_path = download_arxiv_data()
    if not csv_path:
        print("Failed to download dataset")
        return
    
    # Preprocess data
    df, mlb, y_encoded = preprocess_data(csv_path)
    
    # Create text vectorizer using abstracts
    vectorizer, X_vectorized = create_text_vectorizer(df['abstracts'].fillna(''))
    
    # Split data
    data_splits, label_splits = split_data(X_vectorized, y_encoded)
    
    # Save artifacts
    save_preprocessing_artifacts(vectorizer, mlb, data_splits, label_splits)
    
    # Save processed dataframe for recommendations
    df_processed = df[['titles', 'abstracts', 'filtered_terms']].copy()
    df_processed.to_pickle("models/processed_papers.pkl")
    
    print("Data preprocessing completed successfully!")
    return True

if __name__ == "__main__":
    main()