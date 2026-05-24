import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_raw_data(filepath):
    """Tahap Memuat Dataset Mentah"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Berkas mentah tidak ditemukan di: {filepath}")
    # Tambahkan low_memory=False untuk meredam DtypeWarning dari Pandas
    return pd.read_csv(filepath, low_memory=False)

def automated_preprocessing_pipeline(df):
    """
    Fungsi Preprocessing Terotomatisasi (Versi Perbaikan Eror Karakter Kotor)
    """
    # 1. Menghapus Data Duplikat
    df = df.drop_duplicates().copy()
    
    # 2. Definisikan kolom-kolom numerik utama yang akan digunakan
    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 'Credit_Utilization_Ratio']
    
    # 3. PEMBERSIHAN TOTAL: Hapus karakter '_' dan paksa menjadi numerik untuk semua num_cols
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Menghapus baris yang tidak memiliki target 'Credit_Score'
    df = df.dropna(subset=['Credit_Score'])
    
    # 5. Seleksi Fitur (Feature Selection)
    features = num_cols + ['Credit_Score']
    df = df[features].copy()
    
    # 6. Menangani Data Kosong (Missing Values) via Imputasi Median
    # Sekarang aman karena semua kolom dijamin bertipe numerik murni
    for col in num_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        
    # 7. Encoding Data Kategorikal pada Target
    le = LabelEncoder()
    df['Credit_Score'] = le.fit_transform(df['Credit_Score'])
    
    # 8. Standarisasi Fitur Numerik
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
    input_path = "namadataset_raw/train.csv"
    output_directory = "preprocessing/namadataset_preprocessing"
    output_name = "credit_score_clean.csv"
    
    print("[INFO] Memulai otomatisasi preprocessing data (Kriteria Skilled)...")
    
    raw_data = load_raw_data(input_path)
    clean_data = automated_preprocessing_pipeline(raw_data)
    save_clean_dataset(clean_data, output_directory, output_name)
    
    print("[INFO] Proses otomatisasi selesai dijalankan dengan aman.")