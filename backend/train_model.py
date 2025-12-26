import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- FUNGSI UNTUK MEMBERSIHKAN DATA ---

def clean_numeric_value(value_str):
    """
    Membersihkan string angka yang menggunakan format Indonesia/Eropa.
    
    """
    value_str = str(value_str).strip()
    if not value_str or value_str == '-':
        return np.nan
        
    # Hapus pemisah ribuan ('.') - (regex=False DIHAPUS)
    value_str = value_str.replace('.', '')
    # Ubah pemisah desimal (',') menjadi ('.') - (regex=False DIHAPUS)
    value_str = value_str.replace(',', '.')
    
    try:
        return float(value_str)
    except ValueError:
        return np.nan

def clean_volume_value(value_str):
    """
    Membersihkan string volume yang memiliki 'K' (ribu) atau 'M' (juta).
    
    """
    value_str = str(value_str).strip()
    if not value_str or value_str == '-':
        return np.nan
    
    # Bersihkan format angka standar terlebih dahulu - (regex=False DIHAPUS)
    value_str = value_str.replace('.', '').replace(',', '.')
    
    if value_str.endswith('K'):
        # Pastikan hanya mengambil bagian angka
        return float(value_str[:-1]) * 1_000
    elif value_str.endswith('M'):
        return float(value_str[:-1]) * 1_000_000
    try:
        return float(value_str)
    except ValueError:
        return np.nan

# --- PROSES MEMUAT DAN MEMBERSIHKAN DATA ---

# 1. Muat dataset
# Gunakan sep=';' karena file CSV menggunakan titik koma sebagai pemisah
file_path = 'Data Historis XKMS.csv' 
df = pd.read_csv(file_path, sep=',')


print("--- NAMA KOLOM YANG DIBACA PANDAS ---")
print(df.columns)
print("-------------------------------------")

# 2. Terapkan fungsi pembersihan ke kolom yang relevan
price_cols = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
for col in price_cols:
    df[col] = df[col].apply(clean_numeric_value)

df['Vol.'] = df['Vol.'].apply(clean_volume_value)
# Kode BENAR:
df['Perubahan%'] = df['Perubahan%'].str.replace('%', '', regex=False).apply(clean_numeric_value) / 100.0

# 3. Bersihkan dan konversi kolom Tanggal
# 'dayfirst=True' penting karena format menggunakan DD/MM/YYYY
df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)

# 4. Urutkan data dari TERLAMA ke TERBARU (WAJIB!)
df = df.sort_values(by='Tanggal', ascending=True)

# 5. Atur Tanggal sebagai index (praktik terbaik untuk time-series)
df = df.set_index('Tanggal')

# 6. Tangani nilai kosong (jika ada) setelah konversi
# 'ffill' (forward fill) akan mengisi nilai kosong dengan nilai valid sebelumnya
df = df.ffill()
# 'bfill' (backward fill) untuk mengisi jika ada yg kosong di baris pertama
df = df.bfill()
# Ganti nama kolom Vol. menjadi Volume (WAJIB JIKA KODE BERIKUTNYA MENGHARAPKAN 'Volume')
df = df.rename(columns={'Vol.': 'Volume'})
# Ganti nama kolom 'Perubahan%' menjadi 'Perubahan'
df = df.rename(columns={'Perubahan%': 'Perubahan'}) 

# --- DATASET SIAP PAKAI ---
print("Dataset setelah dibersihkan dan siap dipakai:")
print(df.head())
print("\nInfo data baru:")
df.info()


df = df.sort_values('Tanggal')

df['Close_t-1'] = df['Terakhir'].shift(1)
df['Close_t-2'] = df['Terakhir'].shift(2)
df['Close_t-3'] = df['Terakhir'].shift(3)

df['MA_3'] = df['Terakhir'].rolling(window=3).mean()
df['MA_5'] = df['Terakhir'].rolling(window=5).mean()
df['MA_7'] = df['Terakhir'].rolling(window=7).mean()

df['EMA_3'] = df['Terakhir'].ewm(span=3, adjust=False).mean()
df['EMA_5'] = df['Terakhir'].ewm(span=5, adjust=False).mean()

df['Volatility_5'] = df['Terakhir'].rolling(window=5).std()
df['Volatility_10'] = df['Terakhir'].rolling(window=10).std()

df['Price_Range'] = df['Tertinggi'] - df['Terendah']
df['Price_Range_Pct'] = (df['Tertinggi'] - df['Terendah']) / df['Terendah']

df['High_Low_Avg'] = (df['Tertinggi'] + df['Terendah']) / 2
df['Open_Close_Diff'] = df['Pembukaan'] - df['Close_t-1']

df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

df['RSI'] = 100 - (100 / (1 + (df['Terakhir'].diff().clip(lower=0).rolling(14).mean() /
                              (-df['Terakhir'].diff().clip(upper=0).rolling(14).mean()))))

df = df.dropna()

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

X = df[feature_cols]
y = df['Terakhir']

scaler = StandardScaler()


print("-" * 30)
print(f"Shape dari X sebelum scaling: {X.shape}")
print(f"Jumlah sampel (baris) di X: {len(X)}")
print("-" * 30)

X_scaled = scaler.fit_transform(X)

#  TAMPILKAN HASIL STANDARD SCALER
print("-" * 40)
print("MENAMPILKAN HASIL STANDARD SCALER")
print("-" * 40)

df_scaled_preview = pd.DataFrame(
    X_scaled,
    columns=[f"{col}_scaled" for col in feature_cols]
)

print("\n=== 10 Baris Pertama Hasil StandardScaler ===")
print(df_scaled_preview.head(10))

print("\n=== Statistik Data yang Sudah Distandarisasi ===")
print(df_scaled_preview.describe().round(4))

split_index = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split_index]
X_test  = X_scaled[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

param_grid = {
   'n_estimators': [200, 300],
    'max_depth': [15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print("Training model with GridSearchCV...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Evaluation Metrics ===")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== Top 10 Feature Importances ===")
print(feature_importance.head(10))

os.makedirs('model', exist_ok=True)

joblib.dump(best_model, 'model_reksadana_rf_final.pkl')
joblib.dump(scaler, 'scaler_reksadana_rf_final.pkl')

print("\n✅ Model and scaler saved successfully!")
