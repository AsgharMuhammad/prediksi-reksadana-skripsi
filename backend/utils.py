import numpy as np
import pandas as pd

def parse_volume(vol_str):
    vol_str = vol_str.replace(',', '.').replace('K', 'e3').replace('M', 'e6')
    try:
        return eval(vol_str)
    except:
        return np.nan

def preprocess_data(df):
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Volume'] = df['Vol.'].apply(parse_volume)
    df['Perubahan'] = df['Perubahan%'].str.replace('%', '', regex=False).str.replace(',', '.').astype(float)
    df = df.drop(columns=['Vol.', 'Perubahan%'])
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
    df = df[df['Terakhir'] < 10]

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

    return X, y
