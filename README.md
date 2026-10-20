# Sesi 1 – Prediksi Invoice Telat Bayar dengan Deep Learning (MLP)

Studi kasus turunan dari materi **"Make Sense of Data with Analysis and AI"** (Andria Arisal, BRIN).

## Dari materi pemateri

- Kasus PT Maju Bersama; demo klasifikasi invoice telat bayar (due date + 30 hari).
- Late rate ± 13%; model demo: akurasi ± 89% tetapi recall 0% dan ROC ± 0,6.
- Penekanan: data leakage, data tidak seimbang, definisi target, evaluasi jujur, AI hanya membantu.

## Pengembangan dari saya (bukan dari pemateri)

- Dataset **sintetik buatan sendiri** (dataset asli pemateri tidak tersedia).
- Fitur riwayat customer _point-in-time_, split berdasarkan waktu.
- Model MLP (PyTorch) + pembanding LogReg dan Random Forest.

## Cara menjalankan

```bash
pip install -r requirements.txt
python src/01_generate_data.py                # buat dataset/invoices_synthetic.csv
PYTHONPATH=src python src/02_baselines.py     # baseline ML klasik
PYTHONPATH=src python src/03_mlp_pytorch.py   # deep learning (MLP)
```

(Jalankan dari folder `Session-01-Data-Analysis-AI`. Di Windows PowerShell: `$env:PYTHONPATH="src"` dulu.)
