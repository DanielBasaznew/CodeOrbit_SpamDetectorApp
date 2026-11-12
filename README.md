# CodeOrbit SMS Spam Detector App

## Project Overview & Purpose
An end-to-end machine learning classification system and interactive web application that detects whether an SMS message is **spam** or **not spam (ham)** with calibrated confidence scores. Trained on the UCI SMS Spam Collection dataset (`data/SMSSpamCollection`), serialized with `joblib`, and served through both a **Streamlit** web interface (`app.py`) and an interactive CLI (`cli.py`).

---

## How It Works (TF-IDF + Multinomial Naive Bayes & Streamlit)
1. **Training & Serialization (`train.py`):**
   - Loads `data/SMSSpamCollection` (`5,572` raw rows), removes missing values and duplicates (`5,169` unique messages: `87.37%` ham, `12.63%` spam) to prevent train-test data leakage.
   - Splits data 80/20 with stratification (`4,135` train / `1,034` test, `random_state=42`).
   - Fits a scikit-learn `Pipeline` pairing `TfidfVectorizer(stop_words="english", lowercase=True)` with `MultinomialNB()`, achieving **96.71% test accuracy** and **1.0000 spam precision**, then saves the artifact to `spam_model.joblib`.
2. **Inference Module (`predict.py`):**
   - Lazy-loads `spam_model.joblib` once per session (`get_model()`), guards against empty/whitespace-only inputs, and returns `(label, confidence)` from `predict_proba()`.
3. **Streamlit Web App (`app.py`):**
   - **Data Input:** `st.text_area` paired with preset examples managed cleanly via `st.session_state` and `st.selectbox`.
   - **Cached Prediction:** Uses `@st.cache_resource` to keep `spam_model.joblib` in memory across reruns and displays a friendly error via `st.stop()` if the model artifact is missing.
   - **Output Display:** Renders a **Spam** (`>= 50%`) or **Not spam** status banner, confidence `st.metric`, and spam probability `st.progress` bar.
4. **Terminal CLI (`cli.py`):**
   - Provides a lightweight interactive REPL loop powered by `predict_message()` with clean `quit`/`exit`/EOF handling.

---

## Setup & How to Run

1. Navigate into the project directory:
   ```bash
   cd CodeOrbit_SpamDetectorApp
   ```
2. Activate the virtual environment (or create one and install dependencies):
   - **Windows PowerShell:**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Git Bash / Linux / macOS:**
     ```bash
     source .venv/Scripts/activate
     ```
   *(Fresh setup: `pip install -r requirements.txt`)*
3. *(Optional)* Retrain and serialize the model (`spam_model.joblib` is included):
   ```bash
   python train.py
   ```
4. Launch the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
5. Or run the inference demo, interactive CLI, or test suite directly:
   ```bash
   python predict.py
   python cli.py
   python test_spam.py
   ```

---

## Example Output

### Model Training (`python train.py`)
```text
============================================================
  SMS SPAM DATASET SUMMARY
============================================================
Raw rows: 5572 | Deduplicated rows: 5169

Class Distribution:
  - ham  : 4516 (87.37%)
  - spam : 653 (12.63%)

============================================================
  MODEL EVALUATION (MULTINOMIAL NAIVE BAYES)
============================================================
Train samples: 4135 | Test samples: 1034
Test Accuracy: 96.71%

Classification Report:
              precision    recall  f1-score   support

     ham (0)     0.9637    1.0000    0.9815       903
    spam (1)     1.0000    0.7405    0.8509       131

    accuracy                         0.9671      1034
   macro avg     0.9819    0.8702    0.9162      1034
weighted avg     0.9683    0.9671    0.9650      1034

Confusion Matrix (Rows: True [ham, spam] | Cols: Pred [ham, spam]):
[[903   0]
 [ 34  97]]
============================================================

Saved trained pipeline to 'spam_model.joblib'.
```

### Batch Inference Demo (`python predict.py`)
```text
====================================================================
  SPAM DETECTOR INFERENCE DEMO (LOADED FROM spam_model.joblib)
====================================================================
WINNER!! You have won a free prize, ca...  -> spam                       (85.34%)
Hey, are we still meeting for lunch?       -> not spam                   (99.66%)
Free entry in a weekly competition, te...  -> spam                       (76.73%)
I'm free tonight                           -> not spam                   (94.77%)
Your parcel is waiting, click here         -> not spam                   (82.53%)
'   '                                      -> invalid input (empty message) (0.00%)
''                                         -> invalid input (empty message) (0.00%)
====================================================================
```
