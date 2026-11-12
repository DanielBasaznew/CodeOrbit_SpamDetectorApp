import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_and_evaluate(
    data_path: str = "data/SMSSpamCollection",
    model_output_path: str = "spam_model.joblib",
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    # 1. Load tab-separated SMS data
    df = pd.read_csv(data_path, sep="\t", names=["label", "message"])
    initial_count = len(df)

    # 2. Clean empty rows and duplicates to prevent train-test leakage
    df = df.dropna(subset=["label", "message"]).drop_duplicates(subset=["message"]).copy()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    print("=" * 60)
    print("  SMS SPAM DATASET SUMMARY")
    print("=" * 60)
    print(f"Raw rows: {initial_count} | Deduplicated rows: {len(df)}")
    print("\nClass Distribution:")
    for lbl, count in df["label"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  - {lbl:<5}: {count} ({pct:.2f}%)")

    # 3. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"],
    )

    # 4. Build and train scikit-learn Pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("clf", MultinomialNB()),
    ])
    model.fit(X_train, y_train)

    # 5. Evaluate on holdout test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["ham (0)", "spam (1)"], digits=4
    )

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION (MULTINOMIAL NAIVE BAYES)")
    print("=" * 60)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (Rows: True [ham, spam] | Cols: Pred [ham, spam]):")
    print(cm)
    print("=" * 60)

    # 6. Serialize trained pipeline to disk
    joblib.dump(model, model_output_path)
    print(f"\nSaved trained pipeline to '{model_output_path}'.")


if __name__ == "__main__":
    train_and_evaluate()