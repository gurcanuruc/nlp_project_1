"""Savunma deneyi G1: klasik baseline (TF-IDF + lojistik regresyon) vs LLM.

Kullanım:  python exp_g1_baseline.py
Gerekli:   pip install scikit-learn
- Egitim: IMDB train split'ten 5000 ornek (LLM'e hic gosterilmedi)
- Test:   ayni 100 test ornegi (shuffle seed=42, ilk 100; metin ilk 1000 karakter)
- Cikti:  accuracy ve F1 (pozitif sinif); LLM: %97 (T=0), F1 0.967
"""
import time
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000))
test = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(100))

Xtr, ytr = [t[:1000] for t in train["text"]], train["label"]
Xte, yte = [t[:1000] for t in test["text"]], test["label"]

t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = LogisticRegression(max_iter=1000)
clf.fit(vec.fit_transform(Xtr), ytr)
train_s = time.time() - t0

t0 = time.time()
pred = clf.predict(vec.transform(Xte))
pred_ms = (time.time() - t0) / len(Xte) * 1000

print(f"TF-IDF + LogReg  accuracy={accuracy_score(yte, pred):.3f}  F1(pos)={f1_score(yte, pred):.3f}")
print(f"egitim suresi: {train_s:.1f} s | tahmin: {pred_ms:.2f} ms/ornek")
print("LLM (qwen2.5:3b, T=0): accuracy=0.970  F1(pos)=0.967")
print("Not: baseline sadece duygu tahmini yapar, JSON/entity cikarmaz.")