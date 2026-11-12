import os
import joblib
import streamlit as st

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

MODEL_PATH = "spam_model.joblib"


# ------------------------------------------------------------------------------
# 2. Cached Model Loader & Missing-File Guard
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    """Loads the serialized TF-IDF + Naive Bayes pipeline once across reruns."""
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error("Model file (`spam_model.joblib`) not found. Run `python train.py` first.")
    st.stop()

model = load_model(MODEL_PATH)

# ------------------------------------------------------------------------------
# 3. Header & Preset Examples (Session State)
# ------------------------------------------------------------------------------
st.title("📩 SMS Spam Detector")
st.write("Paste an SMS message below or pick an example to check if it looks like spam.")

EXAMPLES = {
    "Custom message (type your own)": "",
    "🚨 Spam: Prize Winner": "WINNER!! You have won a free prize, call now to claim your reward!",
    "🚨 Spam: Weekly Competition": "Free entry in a weekly competition, text WIN to 80086 now. T&Cs apply.",
    "✅ Ham: Lunch Plans": "Hey, are we still meeting for lunch at 12:30?",
    "✅ Ham: Keyword Trap ('free')": "I'm free tonight if you want to grab coffee.",
    "⚠️ Edge Case: Delivery Link": "Your parcel is waiting, click here to reschedule delivery.",
}

if "message_input" not in st.session_state:
    st.session_state.message_input = ""


def on_example_change():
    selected = st.session_state.example_selector
    st.session_state.message_input = EXAMPLES[selected]


st.selectbox(
    "Try a sample message:",
    options=list(EXAMPLES.keys()),
    key="example_selector",
    on_change=on_example_change,
)

# ------------------------------------------------------------------------------
# 4. Data Input
# ------------------------------------------------------------------------------
message = st.text_area(
    "SMS Message Text:",
    key="message_input",
    height=130,
    placeholder="Type or paste an SMS message here...",
)

# ------------------------------------------------------------------------------
# 5. Model Prediction & Output Display
# ------------------------------------------------------------------------------
if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        proba = model.predict_proba([message.strip()])[0]
        ham_prob = float(proba[0])
        spam_prob = float(proba[1])
        is_spam = spam_prob >= 0.5
        confidence = spam_prob if is_spam else ham_prob

        st.subheader("Prediction Result")
        col1, col2 = st.columns([2, 1])

        with col1:
            if is_spam:
                st.error("🚨 Spam")
            else:
                st.success("✅ Not spam")

        with col2:
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

        st.caption(f"Estimated Spam Probability: {spam_prob * 100:.2f}%")
        st.progress(spam_prob)

# ------------------------------------------------------------------------------
# 6. Architecture Explanation Expander
# ------------------------------------------------------------------------------
with st.expander("ℹ️ How it works"):
    st.markdown(
        """
        1. **TF-IDF Vectorization:** `TfidfVectorizer` removes common English stop words and converts your text into numerical term-importance weights.
        2. **Naive Bayes Classification:** `MultinomialNB` estimates the posterior probability that the message belongs to **spam** vs. **not spam (ham)**.
        3. **Decision Rule:** If the estimated spam probability is **50% or higher**, the message is flagged as **Spam**; otherwise, it is marked **Not spam**.
        """
    )