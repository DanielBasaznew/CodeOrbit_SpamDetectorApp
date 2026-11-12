from predict import predict_message


def main() -> None:
    print("=" * 60)
    print("  SMS SPAM DETECTOR — INTERACTIVE CLI")
    print("  Type a message to classify (or 'quit' to exit).")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter SMS > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting SMS Spam Detector CLI.")
            break

        if user_input.strip().lower() in {"quit", "exit"}:
            print("Exiting SMS Spam Detector CLI.")
            break

        label, confidence = predict_message(user_input)
        if label.startswith("invalid"):
            print("[Warning] Please enter a non-empty message.")
        else:
            print(f"Result: {label.upper()} (Confidence: {confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()