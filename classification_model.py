import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, hamming_loss
import matplotlib.pyplot as plt

def create_mlp_model(input_dim, output_dim, hidden_dims=[512, 256], dropout_rate=0.3):
    """Create Multi-Layer Perceptron model for multi-label classification"""
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(hidden_dims[0], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Second hidden layer
        layers.Dense(hidden_dims[1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer with sigmoid for multi-label classification
        layers.Dense(output_dim, activation='sigmoid')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """Compile the model with appropriate optimizer and loss"""
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # for multi-label classification
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def create_callbacks(patience=10):
    """Create training callbacks"""
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def train_model():
    """Train the classification model"""
    print("Loading preprocessed data...")
    
    # Load data splits
    with open("models/data_splits.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_train = data['X_train'].toarray()  # Convert sparse to dense
    X_val = data['X_val'].toarray()
    X_test = data['X_test'].toarray()
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    
    model = create_mlp_model(input_dim, output_dim)
    model = compile_model(model)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model = keras.models.load_model('models/best_model.h5')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Binary Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")
    
    # Generate predictions for detailed evaluation
    y_pred = model.predict(X_test, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate hamming loss
    hamming = hamming_loss(y_test, y_pred_binary)
    print(f"Hamming Loss: {hamming:.4f}")
    
    # Save final model
    model.save('models/model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    # Load vocabulary for detailed evaluation
    with open("models/vocab.pkl", "rb") as f:
        mlb = pickle.load(f)
    
    # Print classification report for most frequent classes
    frequent_classes = np.sum(y_test, axis=0).argsort()[-10:][::-1]
    
    print("\nClassification Report for Top 10 Most Frequent Classes:")
    for i in frequent_classes:
        class_name = mlb.classes_[i]
        precision = np.sum((y_pred_binary[:, i] == 1) & (y_test[:, i] == 1)) / max(np.sum(y_pred_binary[:, i]), 1)
        recall = np.sum((y_pred_binary[:, i] == 1) & (y_test[:, i] == 1)) / max(np.sum(y_test[:, i]), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        support = np.sum(y_test[:, i])
        
        print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")
    
    print("\nTraining completed successfully!")
    return model, history

def plot_training_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Binary Accuracy
    axes[0, 1].plot(history.history['binary_accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Binary Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training history plot saved to models/training_history.png")

def load_trained_model():
    """Load trained model and preprocessing artifacts"""
    
    # Load model
    model = keras.models.load_model('models/model.h5')
    
    # Load vectorizer config
    with open("models/text_vectorizer_config.pkl", "rb") as f:
        vectorizer_config = pickle.load(f)
    
    # Recreate vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        vocabulary=vectorizer_config['vocabulary_'],
        max_features=vectorizer_config['max_features'],
        ngram_range=vectorizer_config['ngram_range'],
        stop_words=vectorizer_config['stop_words'],
        lowercase=vectorizer_config['lowercase'],
        min_df=vectorizer_config['min_df'],
        max_df=vectorizer_config['max_df']
    )
    
    # Set the idf values
    vectorizer.idf_ = vectorizer_config['idf_']
    
    # Load multilabel binarizer
    with open("models/vocab.pkl", "rb") as f:
        mlb = pickle.load(f)
    
    return model, vectorizer, mlb

def predict_categories(text, model=None, vectorizer=None, mlb=None, threshold=0.3):
    """Predict categories for given text"""
    
    if model is None or vectorizer is None or mlb is None:
        model, vectorizer, mlb = load_trained_model()
    
    # Vectorize text
    X = vectorizer.transform([text])
    X_dense = X.toarray()
    
    # Predict
    predictions = model.predict(X_dense, verbose=0)[0]
    
    # Apply threshold
    predicted_indices = np.where(predictions > threshold)[0]
    predicted_categories = mlb.classes_[predicted_indices]
    predicted_scores = predictions[predicted_indices]
    
    # Sort by score
    sorted_results = sorted(zip(predicted_categories, predicted_scores), 
                          key=lambda x: x[1], reverse=True)
    
    return sorted_results

def main():
    """Main training pipeline"""
    
    # Check if preprocessing is done
    if not os.path.exists("models/data_splits.pkl"):
        print("Preprocessed data not found. Running preprocessing first...")
        from data_preprocessing import main as preprocess_main
        if not preprocess_main():
            print("Preprocessing failed!")
            return
    
    # Train model
    model, history = train_model()
    
    # Test prediction
    print("\nTesting prediction with sample text...")
    sample_text = """
    This paper presents a comprehensive study on the application of deep learning 
    techniques for natural language processing tasks. We propose a novel transformer 
    architecture that achieves state-of-the-art performance on text classification 
    and sentiment analysis benchmarks.
    """
    
    predictions = predict_categories(sample_text)
    print("Sample predictions:")
    for category, score in predictions:
        print(f"  {category}: {score:.3f}")

if __name__ == "__main__":
    main()