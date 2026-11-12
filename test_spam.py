from predict import predict_message

test_cases = [
    ("Obvious Spam", "WINNER!! You have won a free prize, call now"),
    ("Obvious Ham", "Hey, are we still meeting for lunch?"),
    ("Promotional Shortcode", "Free entry in a weekly competition, text WIN to 80086"),
    ("Keyword Trap ('free')", "I'm free tonight"),
    ("Borderline Phishing", "Your parcel is waiting, click here"),
    ("Empty String", ""),
    ("Whitespace Only", "     "),
    ("Very Long Ham", "Hey just checking in to see if you finished the project report and whether we should review the slides together tomorrow morning before the team meeting starts at nine. " * 4),
    ("Emoji / Non-English", "🎉🔥🚀 ሰላም እንዴት ነህ?"),
]

print("=" * 78)
print("  HOUR 4 SPAM DETECTOR TEST MATRIX (LOADED FROM spam_model.joblib)")
print("=" * 78)

for case_name, text in test_cases:
    label, confidence = predict_message(text)
    preview = repr(text) if not text.strip() else (text[:35] + "..." if len(text) > 38 else text)
    print(f"{case_name:<23} | {preview:<38} | {label:<10} ({confidence * 100:.2f}%)")

print("=" * 78)