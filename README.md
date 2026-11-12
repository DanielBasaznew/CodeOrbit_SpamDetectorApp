# CodeOrbit SMS Spam Detector (Task 6 — Complete)

An end-to-end machine learning web application that classifies SMS text messages as **spam** or **not spam (ham)** with calibrated confidence scores.

## App Overview
The Streamlit web interface (`app.py`) delivers all three core application layers:
1. **Data Input:** A text box (`st.text_area`) paired with a preset example selector (`st.selectbox`) for instant one-click testing.
2. **Model Prediction:** Validates non-empty input and evaluates the message against the cached TF-IDF + Multinomial Naive Bayes pipeline (`spam_model.joblib`).
3. **Output Display:** Displays a clear **🚨 Spam** or **✅ Not spam** status banner, confidence percentage (`st.metric`), and probability bar (`st.progress`).

![SMS Spam Detector App Screenshot](app_screenshot.png)

---

## Install & Run Steps

1. Clone the repository and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows Git Bash