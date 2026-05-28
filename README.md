# Movie-Sentiment-Analyzer
NLP sentiment analyzer for movie reviews using TF-IDF and an ensemble of Naive Bayes + Logistic Regression.

# 🎬 Movie Sentiment Analyzer

A desktop NLP application that classifies movie reviews as positive or negative using an ensemble machine learning model — built with Python, scikit-learn, and Tkinter.

## 📸 Preview

![Movie-Sentiment-Analyzer](screenshotMOVE.png)

## 🚀 Features

- Analyzes any movie review text and predicts sentiment in real time
- Ensemble classifier combining **Multinomial Naive Bayes** and **Logistic Regression**
- TF-IDF vectorization with 20,000 features and unigram/bigram support
- Negation-aware preprocessing — keeps words like "not", "never", "hardly"
- Confidence score with visual progress bar
- Review history log
- Model Stats tab showing accuracy, precision, recall, and F1-score
- Charts tab with confusion matrix, confidence distribution, top keywords, and review length distribution
- Trains in a background thread so the UI stays responsive

## 🛠️ Built With

- **Python 3**
- **NLTK** — dataset, tokenization, lemmatization, stopwords
- **scikit-learn** — TF-IDF, Logistic Regression, Naive Bayes, VotingClassifier
- **Matplotlib** — embedded charts
- **Tkinter** — GUI

## 📊 Model

| Metric | Value |
|--------|-------|
| Architecture | Ensemble (Naive Bayes + Logistic Regression) |
| Vectorization | TF-IDF · 20,000 features · unigrams & bigrams |
| Dataset | NLTK movie_reviews · 2,000 reviews · 80/20 split |

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/VelosoMiguel/movie-sentiment-analyzer.git
   cd movie-sentiment-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install nltk scikit-learn matplotlib numpy
   ```

3. Run the app:
   ```bash
   python Movie_Analyze_Review.py
   ```

> The app downloads the NLTK datasets automatically on first run.

## 🔮 Future Improvements

- [ ] Support for custom datasets beyond NLTK movie reviews
- [ ] Export results to CSV
- [ ] ROC curve chart
- [ ] Fine-tune with transformer models (BERT)

## 👤 Author

**Miguel Veloso**  
[GitHub](https://github.com/VelosoMiguel) · [LinkedIn](https://www.linkedin.com/in/miguel-veloso-91355b372/)
