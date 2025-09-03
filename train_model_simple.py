import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, jaccard_score
import time

print("Training Research Paper Classification Model")
print("=" * 50)

def train_classification_model():
    """Train a classification model"""
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    try:
        with open('models/data_splits.pkl', 'rb') as f:
            data_splits = pickle.load(f)
        
        X_train = data_splits['X_train']
        X_val = data_splits['X_val'] 
        X_test = data_splits['X_test']
        y_train = data_splits['y_train']
        y_val = data_splits['y_val']
        y_test = data_splits['y_test']
        
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of categories: {y_train.shape[1]}")
        
    except FileNotFoundError:
        print("ERROR: Preprocessed data not found. Please run data_preprocessing.py first.")
        return
    
    # Train MultiOutput Logistic Regression
    print("\nTraining Multi-Output Logistic Regression Model...")
    print("This is faster than deep learning but still very effective")
    
    start_time = time.time()
    
    # Use LogisticRegression with MultiOutputClassifier for multi-label
    base_classifier = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        solver='liblinear',
        C=1.0
    )
    
    model = MultiOutputClassifier(base_classifier, n_jobs=-1)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\nEvaluating Model Performance...")
    
    y_val_pred = model.predict(X_val)
    hamming = hamming_loss(y_val, y_val_pred)
    jaccard = jaccard_score(y_val, y_val_pred, average='macro')
    
    print(f"Validation Metrics:")
    print(f"   - Hamming Loss: {hamming:.4f} (lower is better)")
    print(f"   - Jaccard Score: {jaccard:.4f} (higher is better)")
    print(f"   - Accuracy per sample: {1 - hamming:.4f}")
    
    # Test set evaluation
    print("\nFinal Test Set Evaluation...")
    y_test_pred = model.predict(X_test)
    
    test_hamming = hamming_loss(y_test, y_test_pred)
    test_jaccard = jaccard_score(y_test, y_test_pred, average='macro')
    
    print(f"Test Metrics:")
    print(f"   - Hamming Loss: {test_hamming:.4f}")
    print(f"   - Jaccard Score: {test_jaccard:.4f}")
    print(f"   - Final Accuracy: {1 - test_hamming:.4f}")
    
    # Save the trained model
    print("\nSaving Trained Model...")
    with open('models/trained_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save model performance metrics
    metrics = {
        'training_time': training_time,
        'val_hamming_loss': hamming,
        'val_jaccard_score': jaccard,
        'test_hamming_loss': test_hamming,
        'test_jaccard_score': test_jaccard,
        'model_type': 'MultiOutputLogisticRegression'
    }
    
    with open('models/model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("Model saved successfully!")
    print("   - Model file: models/trained_classifier.pkl")
    print("   - Metrics file: models/model_metrics.pkl")
    
    return model, metrics

def test_model_predictions():
    """Test the trained model with sample predictions"""
    
    print("\nTesting Model with Sample Predictions...")
    
    # Load the trained model and artifacts
    with open('models/trained_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/text_vectorizer_config.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/vocab.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    # Test samples
    test_abstracts = [
        "We propose a novel deep learning approach for computer vision tasks using convolutional neural networks.",
        "This paper presents a new natural language processing model based on transformer architecture.",
        "We develop a reinforcement learning algorithm for autonomous robot navigation.",
        "Our work introduces a statistical machine learning method for analyzing large-scale datasets."
    ]
    
    print("\nSample Predictions:")
    print("-" * 60)
    
    for i, abstract in enumerate(test_abstracts):
        # Vectorize the text
        X_sample = vectorizer.transform([abstract])
        
        # Get prediction probabilities
        pred_proba = model.predict_proba(X_sample)[0]
        
        # Get predictions (using 0.3 threshold)
        pred_binary = (pred_proba > 0.3).astype(int)
        
        # Convert back to category names
        predicted_categories = mlb.inverse_transform([pred_binary])
        
        print(f"\nTest {i+1}:")
        print(f"Abstract: {abstract[:60]}...")
        print(f"Predicted: {', '.join(predicted_categories[0]) if predicted_categories[0] else 'None above threshold'}")
        
        # Show top 3 probabilities
        top_indices = np.argsort(pred_proba)[::-1][:3]
        print("Top 3 probabilities:")
        for idx in top_indices:
            category = mlb.classes_[idx]
            prob = pred_proba[idx]
            print(f"   - {category}: {prob:.3f}")

if __name__ == "__main__":
    # Train the model
    model, metrics = train_classification_model()
    
    # Test with sample predictions
    test_model_predictions()
    
    print("\nTraining Complete!")
    print("=" * 50)
    print("Your classification model is now trained and ready!")
    print("You can now run the advanced app.py to see ML predictions")