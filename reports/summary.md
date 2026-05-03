# Ringkasan Studi Kasus

Dataset hanya berisi fitur kategorikal (`Brand`, `Category`, `Color`, `Size`, `Material`) dan target `Price`.

Kesimpulan singkat:

- Baseline (mean) sering lebih baik daripada model terlatih ketika tidak ada sinyal.
- Model terkadang memberikan R² negatif => model lebih buruk dari prediksi rata-rata.
- Setelah EDA dan rekayasa fitur, tidak ditemukan sinyal yang cukup untuk memprediksi harga.

Takeaway: Machine learning cannot learn patterns that do not exist in data.