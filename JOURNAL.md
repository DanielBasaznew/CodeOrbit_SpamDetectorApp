# Engineering Journal — Day 8: SMS Spam Detector Model (Task 6, Part 1)

## 1. What I Studied
- **Imbalanced Binary Classification:** Why ~87.37% accuracy on the SMS Spam Collection can be achieved by a useless all-ham classifier, making spam precision and recall the critical evaluation metrics.
- **Scikit-Learn Pipelines:** Chaining `TfidfVectorizer` and `MultinomialNB` into one estimator to prevent train-test data leakage.
- **Probability Estimation & Serialization:** Using `predict_proba()` for confidence reporting and `joblib.dump()` / `joblib.load()` to persist models across sessions.

## 2. Core Notes
1. **Spam vs. Ham:** Binary classification distinguishing unwanted promotional/scam SMS (`1`) from legitimate messages (`0`).
2. **Precision Priority:** In spam filtering, precision (`1.0000` achieved) matters more than recall (`0.7405`) because falsely blocking an urgent real message is worse than letting an occasional spam text through.
3. **Deduplication:** Removing 403 duplicate rows (`5,572 -> 5,169`) prevented identical messages from leaking across train and test splits.
4. **Pipeline Consistency:** Wrapping vectorization and classification inside `Pipeline` guarantees identical preprocessing during training and inference.
5. **Model Persistence:** Serializing `spam_model.joblib` decouples offline training from real-time inference.

## 3. What I Built
- Initialized `CodeOrbit_SpamDetectorApp` with an isolated `.venv` and `data/SMSSpamCollection`.
- Implemented `train.py`: deduplicates SMS messages, runs a stratified 80/20 split (`random_state=42`), trains a `TfidfVectorizer` + `MultinomialNB` pipeline, prints classification metrics, and saves `spam_model.joblib`.
- Implemented `predict.py`: lazy-loads `spam_model.joblib`, validates non-empty string input, and returns `(label, confidence)`.
- Built `test_spam.py` to audit keyword traps, borderline phishing, empty inputs, and non-English/emoji text.

## 4. Tests Run & Empirical Findings
- **Test Accuracy:** `96.71%` (vs. `87.37%` majority-class baseline), with `1.0000` spam precision (903/903 ham preserved) and `0.7405` spam recall (97/131 spam caught).
- **Keyword Trap (`"I'm free tonight"`):** Correctly predicted `not spam` (`94.77%` confidence).
- **Modern Phishing Blindspot (`"Your parcel is waiting, click here"`):** Predicted `not spam` (`82.53%`) because 2012 UCI training data lacks modern parcel-delivery scam vocabulary.
- **Out-of-Vocabulary Input (`"🎉🔥🚀 ሰላም እንዴት ነህ?"`):** Produced an all-zero TF-IDF vector, defaulting to the `87.38%` ham prior without crashing.

## 5. End-to-End Explanation (5 Lines)
1. Raw SMS messages are deduplicated, mapped (`ham=0`, `spam=1`), and split 80/20 with stratification.
2. `TfidfVectorizer` converts text into weighted term-frequency sparse vectors while filtering English stop words.
3. `MultinomialNB` computes posterior class probabilities via `predict_proba()`.
4. `joblib.dump()` serializes the fitted pipeline to `spam_model.joblib`.
5. `predict_message()` loads the saved artifact, validates input, and outputs the predicted class and confidence score.

## 6. What I Don't Understand Yet / Next Steps
- How tuning the probability decision threshold (e.g., lowering from `0.50` to `0.35`) trades precision for higher spam recall.
- Adding n-grams (`ngram_range=(1, 2)`) or `LogisticRegression` to catch multi-word phishing cues like `"click here"`.

## 7. Tomorrow's Plan (Day 9: Task 6, Part 2)
- Build an interactive Streamlit web UI (`app.py`) powered by `predict_message()`.
- Complete the `App` and `How It Works` sections in `README.md`.