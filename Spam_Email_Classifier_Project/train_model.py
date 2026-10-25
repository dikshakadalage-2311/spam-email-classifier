import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "spam.csv")
if not {"label", "message"}.issubset(data.columns):
    raise ValueError("spam.csv needs columns named label and message")
data = data[["label", "message"]].dropna()
data["label"] = data["label"].astype(str).str.strip().str.lower()
data["message"] = data["message"].astype(str).str.strip()
data = data[data["label"].isin(["spam", "ham"]) & (data["message"] != "")]
if data["label"].nunique() != 2 or data["label"].value_counts().min() < 2:
    raise ValueError("Need at least two examples for both spam and ham labels")

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.25,
    random_state=42, stratify=data["label"]
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print(classification_report(y_test, predictions, zero_division=0))
joblib.dump(model, ROOT / "spam_model.joblib")
print("Saved spam_model.joblib")
