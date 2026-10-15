from pathlib import Path
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    """Basic text normalization. TF-IDF handles tokenization and vectorization."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the UCI SMS Spam Collection and place its tab-separated "
            "SMSSpamCollection file at data/SMSSpamCollection."
        )
    # UCI file is usually tab-separated and has no header.
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].astype(str).map(clean_text)
    df = df.drop_duplicates(subset=["label", "message"])
    if df.empty:
        raise ValueError("No valid rows found. Check that the dataset is tab-separated.")
    return df


def main():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=None)
    }

    rows = []
    fitted = {}
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_spam": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
            "recall_spam": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
            "f1_spam": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        }
        rows.append(metrics)
        fitted[name] = pipeline

        report = classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0)
        (REPORT_DIR / f"classification_report_{name.lower().replace(' ', '_')}.txt").write_text(report, encoding="utf-8")

        cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legitimate (ham)", "Spam"],
                    yticklabels=["Legitimate (ham)", "Spam"])
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close()

    results = pd.DataFrame(rows).sort_values("f1_spam", ascending=False)
    results.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    best_name = str(results.iloc[0]["model"])
    joblib.dump(fitted[best_name], MODEL_DIR / "spam_model.joblib")
    (MODEL_DIR / "best_model_name.txt").write_text(best_name, encoding="utf-8")

    print(f"Dataset rows after cleaning: {len(df)}")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model by spam F1-score: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'spam_model.joblib'}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
