import json
import urllib.request


DATASET_URL = "https://raw.githubusercontent.com/turkish-nlp-suite/BeyazPerde-Movie-Reviews/refs/heads/main/butun-fimler/all_movies_reviews.json"


def clean_and_parse_rating(rating_str):
    """'4,0' veya '3.5' gibi virgüllü/noktalı puan string'lerini float'a çevirir."""
    if not rating_str:
        return None
    try:
        # Türkçe '4,0' formatını '4.0' yapıp float'a dönüştürüyoruz
        clean_str = str(rating_str).replace(",", ".")
        return float(clean_str)
    except ValueError:
        return None


def fetch_and_save_dataset():
    """Gelen karmaşık JSON yapısını işler, gürültülü yorumları temizler ve dataset.json oluşturur."""
    print("📥 Veri seti indiriliyor ve işleniyor...")

    try:
        req = urllib.request.Request(
            DATASET_URL, 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        with urllib.request.urlopen(req) as response:
            raw_movies = json.loads(response.read().decode("utf-8"))

        positives = []
        negatives = []

        # Eğer tek bir film nesnesi geldiyse listeye çevir, liste geldiyse direkt dön
        movies_list = raw_movies if isinstance(raw_movies, list) else [raw_movies]

        for movie in movies_list:
            reviews = movie.get("reviews", [])
            
            for item in reviews:
                text = item.get("review", "").strip()
                rating_val = clean_and_parse_rating(item.get("rating"))

                if not text or rating_val is None:
                    continue

                # 1. Spam / Reklam ve Anlamsız Yorum Filtresi
                # WhatsApp reklamı, çok kısa metinler veya 'henüz izlemedim' gibi gürültüleri eliyoruz
                if "WHATSAPP" in text.upper() or len(text) < 15 or "seyretmedim" in text.lower():
                    continue

                # 2. Etiketleme (Duygu Sınıflandırması)
                # 4.0 ve üzeri Positive, 2.0 ve altı Negative
                if rating_val >= 4.0:
                    positives.append({"text": text, "label": "Positive"})
                elif rating_val <= 2.0:
                    negatives.append({"text": text, "label": "Negative"})

        # Dengeli bir veri seti için her iki sınıftan eşit sayıda alıyoruz (örneğin 100'er adet)
        positives_sampled = positives[:100]
        negatives_sampled = negatives[:100]
        
        final_dataset = positives_sampled + negatives_sampled

        if not final_dataset:
            print("⚠️ Uyarı: Uygun veri bulunamadı!")
            return

        # dataset.json olarak kaydet
        with open("dataset.json", "w", encoding="utf-8") as file:
            json.dump(final_dataset, file, ensure_ascii=False, indent=2)

        print(f"Başarılı! Toplam {len(final_dataset)} adet temiz yorum 'dataset.json' dosyasına yazıldı.")
        print(f"   ( Positive: {len(positives_sampled)}, Negative: {len(negatives_sampled)} )")

    except Exception as err:
        print(f"Bir hata oluştu: {err}")
        exit(1)


if __name__ == "__main__":
    fetch_and_save_dataset()