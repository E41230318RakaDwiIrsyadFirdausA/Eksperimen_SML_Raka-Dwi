import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_raw_data(filepath):
    """Tahap Memuat Dataset Mentah"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Berkas mentah tidak ditemukan di: {filepath}")
    return pd.read_csv(filepath)

def automated_preprocessing_pipeline(df):
    """
    Fungsi Preprocessing Terotomatisasi (Sama dengan tahapan Eksperimen Notebook)
    Meliputi: Hapus duplikat, cleansing karakter, imputasi median, seleksi fitur, encoding, dan scaling.
    """
    # 1. Menghapus Data Duplikat
    df = df.drop_duplicates().copy()
    
    # 2. Pembersihan Karakter Kotor & Konversi Tipe Data
    df['Age'] = df['Age'].astype(str).str.replace('_', '')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(str).str.replace('_', '')
    df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce')
    
    # 3. Menghapus baris yang tidak memiliki target 'Credit_Score'
    df = df.dropna(subset=['Credit_Score'])
    
    # 4. Seleksi Fitur (Feature Selection)
    features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_Score']
    df = df[features]
    
    # 5. Menangani Data Kosong (Missing Values) via Imputasi Median
    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 'Credit_Utilization_Ratio']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # 6. Encoding Data Kategorikal pada Target
    le = LabelEncoder()
    df['Credit_Score'] = le.fit_transform(df['Credit_Score'])
    
    # 7. Standarisasi Fitur Numerik
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

def save_clean_dataset(df, output_dir, filename):
    """Tahap Menyimpan Hasil Akhir Preprocessing ke Direktori Tujuan"""
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"[SUCCESS] Dataset siap latih berhasil diperbarui dan disimpan di: {full_path}")

if __name__ == "__main__":
    # Menentukan path lokasi sesuai struktur repositori Eksperimen_SML_Raka
    input_path = "namadataset_raw/train.csv"
    output_directory = "preprocessing/namadataset_preprocessing"
    output_name = "credit_score_clean.csv"
    
    print("[INFO] Memulai otomatisasi preprocessing data (Kriteria Skilled)...")
    
    # Menjalankan pipeline sekuensial
    raw_data = load_raw_data(input_path)
    clean_data = automated_preprocessing_pipeline(raw_data)
    save_clean_dataset(clean_data, output_directory, output_name)
    
    print("[INFO] Proses otomatisasi selesai dijalankan dengan aman.")