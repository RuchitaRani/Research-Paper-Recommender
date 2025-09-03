# Research Paper Classification & Recommendation System

A comprehensive machine learning system that performs two main tasks:

1. **Subject Area Classification**: Multi-label classification of research paper abstracts into academic categories
2. **Paper Recommendation**: Semantic similarity-based recommendation of related research papers

## 🚀 Features

- **Multi-label Text Classification** using Multi-Layer Perceptron (MLP)
- **Semantic Similarity Search** using Sentence-BERT transformers
- **Interactive Web Interface** built with Streamlit
- **Real-time Predictions** with adjustable confidence thresholds
- **Comprehensive Dataset** with ArXiv research papers
- **Model Persistence** for fast loading and reuse

## 🏗️ Architecture

### Classification Component
- **Model**: Multi-Layer Perceptron with dropout regularization
- **Input**: Research paper abstracts (TF-IDF vectorized with bigrams)
- **Output**: Multi-hot encoded category labels
- **Categories**: ArXiv taxonomy (cs.LG, cs.AI, cs.CV, stat.ML, etc.)
- **Performance**: ~99% binary accuracy on validation set

### Recommendation Component  
- **Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Input**: Paper titles or research topics
- **Output**: Ranked list of similar papers with similarity scores
- **Similarity Metric**: Cosine similarity on 384-dimensional embeddings

## 📁 Project Structure

```
research-paper-system/
├── app.py                          # Streamlit web application
├── data_preprocessing.py           # Data loading and preprocessing
├── classification_model.py        # Multi-label classification model
├── recommendation_system.py       # Semantic similarity system
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── data/                          # Dataset storage
│   └── arxiv_data_210930-054931.csv
└── models/                        # Trained models and artifacts
    ├── model.h5                   # Classification model
    ├── text_vectorizer_config.pkl # TF-IDF configuration
    ├── vocab.pkl                  # Category vocabulary
    ├── embeddings.pkl             # Pre-computed embeddings
    ├── sentences.pkl              # Paper titles
    └── sentence_transformer/      # BERT model directory
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone or download the project**
```bash
cd research-paper-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **First-time setup**
   - The system will automatically download the dataset
   - Training models will be created on first run
   - This process may take 10-15 minutes initially

## 📊 Dataset

The system uses the ArXiv research papers dataset from Kaggle:
- **Source**: [ArXiv Dataset on Kaggle](https://www.kaggle.com/datasets/spsayakpaul/arxiv-data-research-papers)
- **Size**: ~56,000 research papers
- **Processed**: ~38,000 papers after filtering and preprocessing
- **Format**: CSV with titles, abstracts, and category terms

### Dataset Preprocessing
- Remove duplicate papers based on titles
- Filter categories appearing less than 2 times
- Convert string category lists to proper list format
- Create TF-IDF vectors from abstracts
- Generate multi-label binary encodings

## 🔧 Usage

### Web Application

1. **Launch the app**: `streamlit run app.py`
2. **Navigate sections**:
   - 🏠 **Home**: Overview and system statistics
   - 🔍 **Classification**: Predict categories from abstracts
   - 💡 **Recommendations**: Find similar papers by title
   - 📊 **System Info**: Technical details and model status

### Classification Usage

1. **Input**: Paste a research paper abstract
2. **Configure**: Set confidence threshold and max categories
3. **Predict**: Get ranked category predictions with confidence scores
4. **Examples**: Use provided sample abstracts for testing

### Recommendation Usage

1. **Input**: Enter a paper title or research topic
2. **Configure**: Set number of recommendations and similarity threshold
3. **Search**: Get ranked similar papers with similarity scores
4. **Examples**: Use provided sample queries for testing

## 🎯 Model Performance

### Classification Model
- **Architecture**: Dense(512) → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Dense(output)
- **Training**: Binary crossentropy loss with Adam optimizer
- **Validation**: Early stopping with learning rate reduction
- **Metrics**: Binary accuracy, precision, recall, hamming loss

### Recommendation Model
- **Embedding Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Embedding Dimension**: 384
- **Similarity Range**: 0.0 (different) to 1.0 (identical)
- **Response Time**: <1 second for real-time recommendations

## 🔬 Technical Details

### Classification Pipeline
1. **Text Preprocessing**: TF-IDF vectorization with unigrams and bigrams
2. **Model Training**: MLP with dropout regularization and early stopping
3. **Multi-label Output**: Sigmoid activation for independent category predictions
4. **Inference**: Real-time prediction with adjustable confidence thresholds

### Recommendation Pipeline
1. **Embedding Generation**: Convert titles to 384-dimensional vectors
2. **Similarity Calculation**: Cosine similarity between query and all papers
3. **Ranking**: Sort by similarity scores in descending order
4. **Filtering**: Apply minimum similarity threshold

## 📈 Customization

### Adding New Categories
1. Update dataset with new category labels
2. Retrain classification model: `python classification_model.py`
3. Update vocabulary in preprocessing pipeline

### Extending Dataset
1. Add new papers to CSV dataset
2. Run preprocessing: `python data_preprocessing.py`
3. Retrain both models as needed

### Model Hyperparameters
- **Classification**: Adjust layer sizes, dropout rates, learning rate in `classification_model.py`
- **Recommendations**: Change embedding model or similarity metric in `recommendation_system.py`

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

2. **Model Loading Errors**
   - Delete `models/` directory and restart app for fresh training
   - Check disk space for model files (~500MB total)

3. **Memory Issues**
   - Reduce batch size in training scripts
   - Use smaller embedding models if needed

4. **Dataset Download Fails**
   - System will create sample dataset automatically
   - Manually download dataset and place in `data/` directory

### Performance Optimization
- **CPU**: Reduce batch size and model complexity
- **Memory**: Clear unused models from memory
- **Speed**: Use GPU for training if available

## 📝 Dependencies

### Core Libraries
- `streamlit>=1.28.0` - Web application framework
- `tensorflow>=2.20.0` - Deep learning for classification
- `torch>=2.0.0` - PyTorch for embeddings
- `sentence-transformers>=2.2.0` - BERT-based embeddings
- `scikit-learn>=1.3.0` - ML utilities and metrics
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computations

### Additional Libraries
- `matplotlib>=3.7.0` - Plotting training curves
- `kagglehub>=0.2.0` - Dataset downloading

## 🔗 References

- [ArXiv Dataset](https://www.kaggle.com/datasets/spsayakpaul/arxiv-data-research-papers)
- [Sentence-BERT](https://www.sbert.net/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)

