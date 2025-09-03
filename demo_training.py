import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, jaccard_score, classification_report
import time

print("Demo Training: Research Paper Classification")
print("=" * 50)

def create_demo_model():
    """Create a working demo with top categories only"""
    
    # Load preprocessed data
    print("\nLoading processed papers...")
    df = pd.read_pickle('models/processed_papers.pkl')
    
    with open('models/vocab.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    with open('models/text_vectorizer_config.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f"Total papers: {len(df)}")
    print(f"Total categories: {len(mlb.classes_)}")
    
    # Focus on top 10 most common categories for demo
    category_counts = {}
    for terms_list in df['filtered_terms']:
        for term in terms_list:
            category_counts[term] = category_counts.get(term, 0) + 1
    
    # Get top 10 categories
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_category_names = [cat[0] for cat in top_categories]
    
    print(f"\nTop 10 categories for demo:")
    for cat, count in top_categories:
        print(f"  {cat}: {count} papers")
    
    # Filter dataset to only include papers with these top categories
    def has_top_category(terms_list):
        return any(term in top_category_names for term in terms_list)
    
    demo_df = df[df['filtered_terms'].apply(has_top_category)].copy()
    print(f"\nFiltered to {len(demo_df)} papers with top categories")
    
    # Create binary labels for top categories only
    demo_labels = []
    for terms_list in demo_df['terms']:
        label_vector = [1 if cat in terms_list else 0 for cat in top_category_names]
        demo_labels.append(label_vector)
    
    demo_labels = np.array(demo_labels)
    print(f"Label matrix shape: {demo_labels.shape}")
    
    # Transform abstracts to TF-IDF
    abstracts = demo_df['abstract'].fillna('').tolist()
    X = vectorizer.transform(abstracts)
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, demo_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train simpler model - one classifier per category
    print(f"\nTraining individual classifiers for each category...")
    
    classifiers = {}
    for i, category in enumerate(top_category_names):
        print(f"Training {category}...")
        
        # Check if we have both classes for this category
        y_cat = y_train[:, i]
        if len(np.unique(y_cat)) < 2:
            print(f"  Skipping {category} - only one class present")
            continue
            
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_cat)
        classifiers[category] = clf
    
    print(f"Successfully trained {len(classifiers)} classifiers")
    
    # Test the models
    print(f"\nTesting classifiers...")
    
    predictions = {}
    for category, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        y_true = y_test[:, top_category_names.index(category)]
        
        accuracy = np.mean(y_pred == y_true)
        predictions[category] = {
            'classifier': clf,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_true': y_true
        }
        
        print(f"  {category}: {accuracy:.3f} accuracy")
    
    # Save demo model
    demo_model = {
        'classifiers': classifiers,
        'categories': top_category_names,
        'vectorizer': vectorizer,
        'predictions': predictions
    }
    
    with open('models/demo_model.pkl', 'wb') as f:
        pickle.dump(demo_model, f)
    
    print(f"\nDemo model saved to models/demo_model.pkl")
    return demo_model

def test_demo_model():
    """Test the demo model with sample predictions"""
    
    print(f"\nTesting Demo Model Predictions...")
    print("-" * 50)
    
    # Load demo model
    with open('models/demo_model.pkl', 'rb') as f:
        demo_model = pickle.load(f)
    
    classifiers = demo_model['classifiers']
    categories = demo_model['categories']
    vectorizer = demo_model['vectorizer']
    
    # Test samples
    test_abstracts = [
        "We propose a novel deep learning approach for computer vision tasks using convolutional neural networks and attention mechanisms for image recognition and object detection.",
        "This paper presents a new natural language processing model based on transformer architecture for machine translation and text understanding with improved performance.",
        "We develop a reinforcement learning algorithm for autonomous robot navigation in complex environments using deep Q-learning and policy gradients.",
        "Our work introduces a statistical machine learning method for analyzing large-scale datasets with applications in data mining and pattern recognition tasks."
    ]
    
    for i, abstract in enumerate(test_abstracts):
        print(f"\nTest {i+1}:")
        print(f"Abstract: {abstract[:80]}...")
        
        # Vectorize
        X_sample = vectorizer.transform([abstract])
        
        # Get predictions from each classifier
        predictions = []
        confidences = []
        
        for category in categories:
            if category in classifiers:
                clf = classifiers[category]
                pred = clf.predict(X_sample)[0]
                prob = clf.predict_proba(X_sample)[0]
                confidence = max(prob)  # Get max probability
                
                if pred == 1:
                    predictions.append(category)
                    confidences.append(confidence)
        
        # Sort by confidence
        if predictions:
            sorted_preds = sorted(zip(predictions, confidences), key=lambda x: x[1], reverse=True)
            print("Predicted categories:")
            for category, confidence in sorted_preds:
                print(f"  - {category}: {confidence:.3f}")
        else:
            print("No categories predicted above threshold")

if __name__ == "__main__":
    # Create demo model
    demo_model = create_demo_model()
    
    # Test demo model
    test_demo_model()
    
    print(f"\n" + "=" * 50)
    print("Demo Training Complete!")
    print("Your system now has a working ML classification model")
    print("You can run the advanced app to see real ML predictions")