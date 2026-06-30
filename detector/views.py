import pickle
import os
from django.shortcuts import render
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml', 'model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'ml', 'vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)


def predict_news(request):
    result = None
    confidence = None
    text = ''
    if request.method == 'POST':
        text = request.POST.get('news_text', '')
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        result = 'Real' if prediction == 1 else 'Fake'
        confidence = round(max(proba) * 100, 2)
    return render(request, 'detector/index.html', {
        'result': result,
        'text': text,
        'confidence': confidence
    })



# Create your views here.
