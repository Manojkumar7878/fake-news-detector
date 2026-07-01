# 📰 Fake News Detector

A machine learning powered web application that analyzes news articles and predicts whether they are **Real** or **Fake** using Natural Language Processing and Logistic Regression.

🔗 **Live Demo:** https://fake-news-detector-1-klnf.onrender.com
🔗 **GitHub:** https://github.com/Manojkumar7878/fake-news-detector

---

## ✨ Features

- 🔍 **Instant Analysis** — paste any news headline or article and get a prediction in real time
- 📊 **Confidence Score** — shows how confident the model is with an animated progress bar
- 🎨 **Clean Responsive UI** — modern interface that works on desktop and mobile
- 🧠 **99% Accuracy** — trained on 44,000+ labeled news articles

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Django |
| Machine Learning | Scikit-learn (Logistic Regression) |
| NLP | NLTK, TF-IDF Vectorization |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS |
| Deployment | Render |

---

## 🧠 How It Works

```
44,000+ News Articles (ISOT Dataset)
            ↓
   Clean text with NLTK
   (lowercase, remove punctuation, remove stopwords)
            ↓
   Convert text to numbers — TF-IDF Vectorizer
            ↓
   Train Logistic Regression model → 99% accuracy
            ↓
   Save model.pkl + vectorizer.pkl
            ↓
   User submits article → Django loads model
            ↓
   Predict Real or Fake + confidence %
            ↓
   Result shown with animated confidence bar
```

---

## 📁 Project Structure

```
fake-news-detector/
├── detector/
│   ├── views.py           # loads model, handles prediction
│   ├── urls.py
│   └── templates/
│       └── detector/
│           └── index.html  # main UI
├── fakenews/
│   ├── settings.py
│   └── urls.py
├── ml/
│   ├── train.py            # training script
│   ├── model.pkl           # trained model
│   └── vectorizer.pkl      # TF-IDF vectorizer
├── manage.py
└── requirements.txt
```

---

## 📊 Model Details

- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF (5,000 max features)
- **Dataset:** ISOT Fake News Dataset (~44,000 articles)
- **Train/Test Split:** 80% / 20%
- **Test Accuracy:** 99%

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Manojkumar7878/fake-news-detector.git
cd fake-news-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run server
python manage.py runserver
```

Visit `http://127.0.0.1:8000`

### Retrain model (optional)
```bash
cd ml
python train.py
```

---

## 👨‍💻 Author

**Manojkumar M**
- GitHub: [@Manojkumar7878](https://github.com/Manojkumar7878)
- LinkedIn: [manojkr2003](https://www.linkedin.com/in/manojkr2003/)
- Email: manokr447@gmail.com
