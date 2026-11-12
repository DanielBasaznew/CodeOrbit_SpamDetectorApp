# Engineering Journal — Day 9: Streamlit Web App & End-to-End Wrap-Up (Task 6, Part 2)

## 1. What I Studied
- **Three App Pillars:** Structuring an end-to-end ML app with visible Data Input (`st.text_area`), Model Prediction (`predict_proba`), and Output Display (`st.error`/`st.success` + `st.metric`).
- **Streamlit Execution Model:** How Streamlit re-executes the entire script from top to bottom on every widget interaction.
- **Caching & State Management:** Using `@st.cache_resource` to load `spam_model.joblib` once in memory and `st.session_state` to populate preset dropdown examples cleanly across reruns.

## 2. Core Notes
1. **App Architecture:** Data input captures text, prediction runs `model.predict_proba()`, and output renders the verdict banner, confidence score, and probability progress bar.
2. **Script Reruns:** Every button click or dropdown change triggers a full top-to-bottom script rerun.
3. **Resource Caching:** `@st.cache_resource` avoids reloading `spam_model.joblib` from disk on every click.
4. **Defensive Guards:** Checking `os.path.exists("spam_model.joblib")` with `st.stop()` and guarding blank inputs with `st.warning()` prevents crashes.
5. **CLI Fallback:** Building `cli.py` around `predict_message()` ensures terminal usability alongside the browser interface.

## 3. What I Built
- Built `app.py` with preset example dropdowns (`st.session_state`), cached `.joblib` loading (`@st.cache_resource`), confidence metrics, progress bar, and an expandable `"How it works"` section.
- Built `cli.py` for interactive terminal inference.
- Captured `app_screenshot.png` and finalized `README.md` with plain-English architecture notes, metrics, and limitations.

## 4. Tests Run & Empirical Findings
- **Preset & Edge-Case UI Tests:** Verified high-confidence spam (`85.34%`), high-confidence ham (`99.66%`), empty/whitespace warnings, emoji/Amharic text (`🎉🔥🚀 ሰላም እንዴት ነህ?` -> `Not spam` without crashing), repeated clicks, and browser refreshes.
- **Missing-Model Test:** Temporarily renamed `spam_model.joblib` to confirm the clean error banner without a Python traceback, then restored the file.
- **Clean-Environment Verification:** Copied project files to a fresh folder (`CodeOrbit_SpamDetector_CleanTest`), created a new `.venv`, ran `pip install -r requirements.txt` and `python train.py`, and reproduced the exact `96.71%` accuracy and `1.0000` spam precision.

## 5. End-to-End Explanation (5 Lines)
1. The user enters an SMS message or selects a preset example in the Streamlit interface.
2. Clicking **Check message** validates non-empty text and passes the string to the cached `Pipeline`.
3. `TfidfVectorizer` transforms the text into numerical term-importance weights.
4. `MultinomialNB` computes posterior probabilities for `ham` and `spam`.
5. The UI displays **Spam** (`>= 50%`) or **Not spam** alongside the confidence percentage and probability bar.

## 6. What I Don't Understand Yet / Next Steps
- Deploying the Streamlit app publicly via Streamlit Community Cloud or Docker.
- Comparing Multinomial Naive Bayes against Logistic Regression or fine-tuned transformer models on modern SMS phishing datasets.

## 7. Tomorrow's Plan (Day 10: Wrap-Up & Submission)
- Locate the submission form link and verify all 6 GitHub repository URLs.
- Record and publish the LinkedIn demo video showcasing the SMS Spam Detector app.
