import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_evulate() -> None:
    """Veri setini okur, modeli eğitir, değerlendirme çıktılarını verir."""


    # 1. Veri Setini Oku
    try:
        with open("dataset.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as err:
        print("'dataset.json' bulunamadı! İlk önce 'fetch_dataset.py' dosyasını çalıştırarak veri setini oluşturun.")
        exit(1)

    X = [item["text"] for item in data]
    y = [item["label"] for item in data]

    # 2. Veriyi Eğitim (%80) ve Test (%20) olarak ikiye böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Boru Hattı (Pipeline) ve Eğitimi
    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    model.fit(X_train, y_train)

    # 4. Modelin Test Raporu
    print("\n--- Model Performans Raporu ---")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Genel Doğruluk Oranı: %{acc * 100:.2f}")

    print("\nDetaylı Metrikler:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # 5. Karmaşıklık Matrisini Hesapla ve Çiz
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Tahmin: Negatif", "Tahmin: Pozitif"],
        yticklabels=["Gerçek: Negatif", "Gerçek: Pozitif"]
    )

    plt.title("Model Karmaşıklık Matrisi (Confusion Matrix)")
    plt.tight_layout()

    # Yazım hatası düzeltildi: confusion_matrix.png
    plt.savefig("confusion_matrix.png", dpi=600)
    print("Grafik 'confusion_matrix.png' olarak kaydedildi!")

    # 6. Eğitilen Modeli Kaydet
    joblib.dump(model, "sentiment_model.pkl")
    print("Model 'sentiment_model.pkl' dosyasına kaydedildi!")


if __name__ == "__main__":
    train_and_evulate()