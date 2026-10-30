# 🎭 Multi-Label Emotion Classification with BERT & ALBERT

A Natural Language Processing (NLP) project that detects complex human emotions in text by comparing baseline logistic regression with advanced Transformer-based architectures.

---

## 🚀 Project Overview
This project addresses the complexity of human emotional expression in written text. Unlike binary sentiment analysis (positive/negative), this model performs **multi-label classification** to detect five distinct emotions: **Joy, Sadness, Fear, Anger, and Surprise**. 

The goal was to build a system capable of understanding that a single sentence can express multiple emotions simultaneously (e.g., a statement can be both "Surprising" and "Sad"). We developed a pipeline to preprocess noisy text data and fine-tune state-of-the-art Transformer models (BERT, ALBERT) to maximize F1 scores and minimize Hamming Loss.

---

## 📊 Dataset Information
The project utilizes the a given dataset, comprising 2,768 English text entries labeled for emotion detection.

### Features
The input data consists of raw text strings which often include "social media noise," such as:
- Emojis and special characters.
- HTML tags and non-standard punctuation.
- Informal grammar and slang.

### Target Labels
The target variable is a multi-hot encoded vector representing five emotions:
- **0**: Anger
- **1**: Fear
- **2**: Joy
- **3**: Sadness
- **4**: Surprise

*Note: Since this is a multi-label problem, a single text input can have multiple `1`s (e.g., `[0, 1, 0, 1, 0]` represents Fear and Sadness).*

---

## 🛠️ Technical Implementation

| Technology | Purpose |
|------------|---------|
| Python | Primary programming language |
| TensorFlow / Keras | Deep learning backend for model training |
| Hugging Face Transformers | Implementation of BERT and ALBERT architectures |
| Scikit-learn | Baseline models (Logistic Regression), TF-IDF, and metrics |
| NLTK & ftfy | Text cleaning, tokenization, and unicode fixing |
| Pandas & NumPy | Data manipulation and vectorization |
| Seaborn | Visualization of class distributions and confusion matrices |

---

## 🔍 Methodology

### Data Preprocessing
To handle the noisy nature of the dataset, we implemented two comparative preprocessing pipelines:
- **Standard Pipeline**: Standard NLTK tokenization and stop-word removal.
- **Custom Pipeline**: A specialized script designed to handle specific artifacts found in the dataset, including emoji translation and HTML decoding, using `ftfy` and regex.

### Model Architecture & Optimization
We implemented and compared three distinct model architectures:

1.  **Baseline (Logistic Regression)**: Used TF-IDF vectorization. Served as a benchmark for performance.
2.  **DistilBERT**: A lighter, faster version of BERT to test Transformer capabilities on the dataset.
3.  **ALBERT (A Lite BERT)**: The final high-performance model. We fine-tuned the pre-trained ALBERT base model, adding a dense output layer with a sigmoid activation function to handle independent probabilities for each emotion class.

---

## 📈 Key Findings

### Model Performance
The deep learning approaches significantly outperformed the statistical baseline.

- **Best Model**: ALBERT (Fine-tuned)
- **Micro F1 Score**: ~70%
- **Macro F1 Score**: ~66%

### Interpretation
- **Threshold Tuning**: We found that standard thresholds (0.5) were suboptimal. Custom threshold tuning was applied to prediction probabilities to minimize Hamming Loss.
- **Complexity Handling**: The Transformer models successfully captured context that TF-IDF missed, particularly in short sentences where the emotion was implied rather than explicitly stated.

---

## 🧠 Machine Learning Pipeline
The project follows a rigorous end-to-end workflow:

1.  **Exploratory Data Analysis (EDA)**: Analyzing label frequency and correlation (e.g., how often "Fear" and "Sadness" appear together).
2.  **Preprocessing**: Cleaning text using custom regex and `ftfy` to fix encoding errors.
3.  **Tokenization**: Converting text into BERT/ALBERT compatible tokens (Input IDs and Attention Masks).
4.  **Training**: Fine-tuning the Transformer models over several epochs with Adam optimization.
5.  **Evaluation**: Generating ROC Curves, calculating Micro/Macro F1 scores, and analyzing Hamming Loss.
6.  **Prediction**: A deployment-ready function that takes raw string input and outputs the detected emotion labels.

---

## 👥 Contributors & Privacy
*This project was completed as part of a university assignment.*

- **Team Lead & Planning**: [Soheil Kafaei Kashefi] - Organized milestones, led team meetings, and managed the project proposal. Name of other teammates are not shared due to their privacy.

## ⚠️ Constraints & Trade-offs
- Selected ALBERT (A Lite BERT) over larger architectures to adhere to a strict 200MB deployment size limit, successfully balancing storage constraints with high performance (70% F1 Score.)
