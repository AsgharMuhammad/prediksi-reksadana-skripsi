from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import traceback

# --- SETUP FASTAPI ---
app = FastAPI(
    title="API Prediksi Harga Reksa Dana",
    description="Prediksi harga reksa dana berbasis Random Forest dengan fitur lag & moving average.",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MUAT MODEL DAN SCALER ---
MODEL_PATH = "model_reksadana_rf_final.pkl"
SCALER_PATH = "scaler_reksadana_rf_final.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model atau scaler: {str(e)}")
    model = None 
    scaler = None 

# --- FUNGSI PEMBERSIHAN DATA  ---

def clean_numeric_value(value_str):
    """
    Membersihkan string angka format Indonesia/Eropa ('1.002,28' -> 1002.28).
    """
    value_str = str(value_str).strip()
    if not value_str or value_str == '-':
        return np.nan
    
    # Hapus titik (pemisah ribuan) dan ganti koma (desimal) dengan titik.
    value_str = value_str.replace('.', '') 
    value_str = value_str.replace(',', '.')
    try:
        return float(value_str)
    except ValueError:
        return np.nan

def clean_volume_value(value_str):
    """
    Membersihkan string volume ('13,32K' -> 13320.0, '5,1M' -> 5100000.0).
    """
    value_str = str(value_str).strip()
    if not value_str or value_str == '-':
        return np.nan
    
    # Bersihkan format angka sebelum mengecek K/M
    value_str = value_str.replace('.', '').replace(',', '.') 
    
    if value_str.endswith('K'):
        return float(value_str[:-1]) * 1_000
    elif value_str.endswith('M'):
        return float(value_str[:-1]) * 1_000_000
    try:
        return float(value_str)
    except ValueError:
        return np.nan

# --- FUNGSI PREPROCESSING FITUR ---

def preprocess_data(df: pd.DataFrame) -> tuple:
    try:
        if "Tanggal" not in df.columns:
            raise ValueError("Kolom 'Tanggal' tidak ditemukan dalam dataset.")

        df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
        df = df.sort_values("Tanggal").dropna(subset=["Tanggal"]).reset_index(drop=True)

        if "Terakhir" not in df.columns:
            raise ValueError("Kolom 'Terakhir' (target) tidak ditemukan dalam dataset.")

        # --- Feature Engineering ---
        # Lag Features
        df['Close_t-1'] = df['Terakhir'].shift(1)
        df['Close_t-2'] = df['Terakhir'].shift(2)
        df['Close_t-3'] = df['Terakhir'].shift(3)

        # Simple Moving Average (SMA)
        df['MA_3'] = df['Terakhir'].rolling(window=3).mean()
        df['MA_5'] = df['Terakhir'].rolling(window=5).mean()
        df['MA_7'] = df['Terakhir'].rolling(window=7).mean()

        # Exponential Moving Average (EMA)
        df['EMA_3'] = df['Terakhir'].ewm(span=3, adjust=False).mean()
        df['EMA_5'] = df['Terakhir'].ewm(span=5, adjust=False).mean()

        # Volatility
        df['Volatility_5'] = df['Terakhir'].rolling(window=5).std()
        df['Volatility_10'] = df['Terakhir'].rolling(window=10).std()

        # Other Features
        df['Perubahan'] = df['Terakhir'].pct_change()
        df['Price_Range'] = df['Tertinggi'] - df['Terendah']
        df['Price_Range_Pct'] = (df['Tertinggi'] - df['Terendah']) / df['Terendah']
        df['High_Low_Avg'] = (df['Tertinggi'] + df['Terendah']) / 2
        df['Open_Close_Diff'] = df['Pembukaan'] - df['Close_t-1']

        # Volume Features
        if 'Volume' in df.columns and df['Volume'].dtype != object: # Cek jika Volume ada dan sudah numerik
            df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        else:
            # Jika Volume tidak ada atau bermasalah, gunakan kolom 0
            df['Volume'] = 0
            df['Volume_MA_3'] = 0
            df['Volume_MA_5'] = 0

        # RSI (Relative Strength Index)
        price_diff = df['Terakhir'].diff()
        gain = price_diff.clip(lower=0)
        loss = -price_diff.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Hapus baris NaN yang muncul akibat feature engineering (shift, rolling)
        df = df.dropna().reset_index(drop=True)
        df = df[df['Terakhir'] >= 500]

        feature_cols = [
            'Close_t-1', 'Close_t-2', 'Close_t-3',
            'Pembukaan', 'Tertinggi', 'Terendah',
            'Volume', 'Perubahan',
            'MA_3', 'MA_5', 'MA_7',
            'EMA_3', 'EMA_5',
            'Volatility_5', 'Volatility_10',
            'Price_Range', 'Price_Range_Pct',
            'High_Low_Avg', 'Open_Close_Diff',
            'Volume_MA_3', 'Volume_MA_5',
            'RSI'
        ]

        # Pastikan semua kolom fitur ada dan DataFrame tidak kosong
        if df.empty:
            raise ValueError("DataFrame kosong setelah pembersihan dan feature engineering.")
            
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Kolom fitur '{col}' hilang setelah preprocessing.")

        X = df[feature_cols]
        y = df["Terakhir"]

        if scaler:
            # Pastikan scaler adalah StandardScaler
            if isinstance(scaler, StandardScaler):
                 X_scaled = scaler.transform(X)
            else:
                 # Jika scaler yang dimuat bukan StandardScaler, fit ulang untuk keamanan
                 X_scaled = StandardScaler().fit_transform(X)
        else:
            raise ValueError("Scaler tidak berhasil dimuat. Tidak bisa melakukan scaling.")

        return df, X_scaled, y, feature_cols
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Kesalahan preprocessing: {str(e)}")

# --- ENDPOINT PREDIKSI ---

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model atau Scaler tidak tersedia. Cek log server.")
        
    try:
        contents = await file.read()
        
        # --- BLOK PEMBACAAN DAN MAPPING ---
        
        # MEMBACA DENGAN DELIMITER KOMA (,) SESUAI PERMINTAAN
        df = pd.read_csv(StringIO(contents.decode('utf-8')), sep=',')
            
        # 1. Pembersihan Nama Kolom Agresif (menghapus spasi, newline, dsb.)
        df.columns = df.columns.str.replace(r'[\r\n\s]+', ' ', regex=True).str.strip()

        # 2. Mapping Nama Kolom (Untuk mengatasi perbedaan Case atau Ejaan)
        column_mapping = {
            'terakhir': 'Terakhir',
            'Last': 'Terakhir',
            'Close': 'Terakhir',
            'open': 'Pembukaan',
            'high': 'Tertinggi',
            'low': 'Terendah',
            'Tgl': 'Tanggal',
            'Vol': 'Vol.',
            'Volume': 'Vol.',
            
        }
        
        # Ganti nama kolom menggunakan mapping yang tersedia
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        
        # --- AKHIR BLOK PEMBACAAN DAN MAPPING ---

        # --- BLOK PEMBERSIHAN DATA ---
        
        # Daftar kolom wajib setelah pemetaan.
        price_cols = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
        for col in price_cols:
            if col not in df.columns:
                # Jika masih error di sini, berarti nama kolom yang benar belum masuk ke column_mapping!
                raise HTTPException(status_code=400, detail=f"Kolom wajib '{col}' tidak ditemukan di file CSV. Pastikan nama kolom 'Terakhir' sudah dipetakan dengan benar.")

            # Lanjutkan pembersihan nilai numerik
            df[col] = df[col].apply(clean_numeric_value)
        
        # 2. Bersihkan kolom Volume ('Vol.' -> 'Volume' yang bersih)
        if 'Vol.' in df.columns:
            df['Volume'] = df['Vol.'].apply(clean_volume_value)
            df = df.drop(columns=['Vol.']) # Hapus kolom 'Vol.' asli
        
        # 3. Hapus kolom persentase perubahan yang lama (jika ada) karena akan dihitung ulang
        if 'Perubahan%' in df.columns:
            df = df.drop(columns=['Perubahan%'])
            
        # 4. Tangani nilai kosong
        df = df.ffill().bfill() 
        # --- AKHIR BLOK PEMBERSIHAN ---
        
        # Lanjutkan ke preprocessing fitur
        df_processed, X_scaled, y_true, feature_cols = preprocess_data(df)

        if len(X_scaled) == 0:
            raise HTTPException(status_code=400, detail="Data tidak cukup untuk prediksi setelah preprocessing (mungkin terlalu sedikit baris).")

        y_pred = model.predict(X_scaled)

        # --- PERHITUNGAN METRIK LENGKAP UNTUK REGRESI ---
        
        # Hitung metrik dasar
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Hitung komponen R-squared, SSR, dan SST
        y_mean = y_true.mean()
        
        # SSR (Sum of Squared Residuals): Jumlahan dari Kuadrat Selisih Aktual vs Prediksi
        SSR = np.sum((y_true - y_pred) ** 2)
        
        # SST (Total Sum of Squares): Jumlahan dari Kuadrat Selisih Aktual vs Rata-rata Aktual
        SST = np.sum((y_true - y_mean) ** 2)
        
        # R-squared
        if SST != 0:
            r_squared = 1 - (SSR / SST)
        else:
            r_squared = np.nan # Menghindari pembagian nol
        # --- AKHIR PERHITUNGAN METRIK ---

        # Siapkan hasil (Mengubah nama kolom menjadi Bahasa Indonesia yang konsisten)
        hasil_df = df_processed[['Tanggal']].copy()
        hasil_df['Aktual'] = y_true.values
        hasil_df['Prediksi'] = y_pred
        hasil_df['Selisih'] = y_true - y_pred
        hasil_df['Selisih_Absolut'] = abs(y_true - y_pred)
        
        # Kolom yang diminta: Selisih Kuadrat (Komponen SSR per baris)
        hasil_df['Selisih_Kuadrat'] = (y_true - y_pred) ** 2 

        # Kolom yang diminta: Selisih Kuadrat Total (Komponen SST per baris)
        hasil_df['Selisih_Kuadrat_Total'] = (y_true - y_mean) ** 2 
        
        hasil_df['Tanggal'] = hasil_df['Tanggal'].dt.strftime('%Y-%m-%d')
        
        # Mengirim total SSR dan SST di bagian evaluasi
        return {
            "evaluasi": {
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "SSR": round(SSR, 4), 
                "SST": round(SST, 4), 
                "R_Squared": round(r_squared, 4) if not np.isnan(r_squared) else None
            },
            "data": hasil_df.to_dict(orient='records')
        }

    except Exception as e:
        print("=== TERJADI ERROR SAAT PREDIKSI ===")
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")


@app.get("/")
def home():
    return {"message": "API Prediksi Harga Reksa Dana - Enhanced version with advanced features 🚀"}
