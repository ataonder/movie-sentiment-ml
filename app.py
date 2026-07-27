import joblib


def load_model():
    """Eğitilmiş modeli diskten yükler."""
    try:
        model = joblib.load("sentiment_model.pkl")
        return model
    except FileNotFoundError:
        print("'sentiment_model.pkl' bulunamadı!")
        print("Lütfen önce 'python train.py' komutunu çalıştırarak modeli eğitin.")
        exit(1)


def main():
    model = load_model()

    print("\n" + "-" * 45)
    print("FILM YORUMU DUYGU ANALIZI SISTEMI (ML)")
    print("(Çıkmak için 'q' yazıp Enter'a basın)")
    print("-" * 45 + "\n")

    while True:
        user_input = input("\nBir film yorumu yazın: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "q":
            print("\nUygulamadan çıkılıyor.")
            break

        prediction = model.predict([user_input])[0]

        if prediction == "Positive":
            emoji = "  Positive (Olumlu)"
        else:
            emoji = "  Negative (Olumsuz)"

        print(f"Tahmin edilen duygu: {emoji}")


if __name__ == "__main__":
    main()