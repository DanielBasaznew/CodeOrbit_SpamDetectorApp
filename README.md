# CodeOrbit SMS Spam Detector (Task 6 — Part 1)

An end-to-end machine learning pipeline that classifies SMS text messages as **spam** or **not spam (ham)** with calibrated probability confidence scores.

## Project Purpose
- Separate model training and evaluation (`train.py`) from fast production inference (`predict.py`) via `joblib` serialization.
- Address class imbalance (~87.4% ham vs. ~12.6% spam) by prioritizing spam precision and recall over raw accuracy alone.

---

## Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Setup:** Download and extract `SMSSpamCollection` into `data/SMSSpamCollection`.
- **Size & Distribution:**
  - Raw rows: `5,572` | Deduplicated rows: `5,169`
  - **Ham (0):** `4,516` messages (`87.37%`)
  - **Spam (1):** `653` messages (`12.63%`)

---

## How to Run

1. Activate the virtual environment:
   ```bash
   source .venv/Scripts/activate  # Windows Git Bash