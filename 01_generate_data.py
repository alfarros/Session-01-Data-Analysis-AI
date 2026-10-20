"""
Tahap 2: Membuat dataset SINTETIK invoice (terinspirasi kasus PT Maju Bersama).

CATATAN PENTING: dataset asli PT Maju Bersama milik pemateri tidak kita punya.
Data ini dibuat sendiri (bukan data asli) untuk keperluan tugas. Angka-angka
(120 customer, ~4.000 invoice, late rate ~13%) mengikuti yang disebut pemateri,
tetapi isi/pola datanya adalah asumsi kita.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_CUSTOMER = 120
N_INVOICE = 4000
START, LAST_ISSUE = pd.Timestamp("2024-07-01"), pd.Timestamp("2026-04-30")
DATA_END = pd.Timestamp("2026-06-30")   # tanggal data "diambil"

# --- 1. Customer: tiap customer punya "risiko telat" laten (tidak terlihat model)
cust = pd.DataFrame({
    "customer_id": np.arange(N_CUSTOMER),
    "segment": rng.choice(["retail", "grosir", "horeca"], N_CUSTOMER, p=[.5, .3, .2]),
    "latent_risk": rng.beta(1.2, 7, N_CUSTOMER),   # kebanyakan rendah, sedikit tinggi
})

# --- 2. Invoice
inv = pd.DataFrame({
    "invoice_id": np.arange(N_INVOICE),
    "customer_id": rng.choice(N_CUSTOMER, N_INVOICE),
    "issue_date": START + pd.to_timedelta(
        rng.integers(0, (LAST_ISSUE - START).days + 1, N_INVOICE), unit="D"),
})
inv = inv.merge(cust, on="customer_id")
inv["amount"] = np.round(rng.lognormal(15.5, 0.8, N_INVOICE), -3)       # Rupiah
inv["n_items"] = rng.integers(1, 8, N_INVOICE)
inv["discount_pct"] = np.round(rng.choice([0, 0, 0, 5, 10, 15], N_INVOICE), 0)
inv["terms_days"] = rng.choice([14, 30, 45], N_INVOICE, p=[.3, .5, .2])
inv["payment_method"] = rng.choice(["transfer", "cash", "giro"], N_INVOICE, p=[.6, .25, .15])
inv["due_date"] = inv["issue_date"] + pd.to_timedelta(inv["terms_days"], unit="D")

# --- 3. Hari terlambat bayar (dipengaruhi risiko laten + fitur invoice)
amt_z = (np.log(inv["amount"]) - 15.5) / 0.8
logit = (-3.75 + 7.0 * inv["latent_risk"] + 0.35 * amt_z
         + 0.04 * inv["discount_pct"] + 0.5 * (inv["payment_method"] == "giro")
         + 0.4 * (inv["terms_days"] == 45))
p_very_late = 1 / (1 + np.exp(-logit))
very_late = rng.random(N_INVOICE) < p_very_late
days_late = np.where(very_late,
                     rng.integers(31, 120, N_INVOICE),      # telat > 30 hari
                     rng.integers(-10, 20, N_INVOICE))      # bayar tepat / sedikit telat
inv["paid_date"] = inv["due_date"] + pd.to_timedelta(days_late, unit="D")

# --- 4. Kolom BOCOR (hanya diketahui SETELAH kejadian) -> harus dibuang saat modeling
inv["days_overdue"] = days_late
inv["late30"] = (days_late > 30).astype(int)                 # TARGET
inv.loc[inv["paid_date"] > DATA_END, "paid_date"] = pd.NaT   # belum dibayar saat data diambil

inv = inv.drop(columns=["latent_risk"]).sort_values("issue_date").reset_index(drop=True)
inv.to_csv("dataset/invoices_synthetic.csv", index=False)
print(f"Tersimpan: {len(inv)} invoice | late rate = {inv['late30'].mean():.1%}")
