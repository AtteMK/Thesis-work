# GEREKLİ KÜTÜPHANELER
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# VERİYİ YÜKLE
df = pd.read_csv("bira_verisi_tum_siniflar.csv")

# VERİ TEMİZLİĞİ
df = df[
    (df["gas_resistance"] > 1000) &
    (df["gas_resistance"] < 1e8) &
    (df["humidity"].between(0, 100)) &
    (df["pressure"].between(900, 1100)) &
    (df["temperature"].between(0, 60))
]

# ÖZELLİKLER VE ETİKETLER
X = df[["gas_resistance", "humidity", "pressure", "temperature"]].values
y = df["abv"].values

# ETİKETLERİ SAYISALA ÇEVİR
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ÖZELLİKLERİ ÖLÇEKLE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# EĞİTİM VE TEST VERİSİNE AYIR
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# MODEL TANIMI
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')
])

# DERLEME
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EĞİTİM
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# DEĞERLENDİRME
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# MODELİ KAYDET
model.save("bira_abv_model.h5")