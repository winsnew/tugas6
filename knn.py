import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# Definisikan dataset
data = pd.DataFrame({
    'X1': [10, 2, 4, 6, 8, 7, 4, 6, 7, 6],
    'X2': [7, 3, 2, 4, 6, 5, 3, 3, 4, 3],
    'y': ['A', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A']
})

# Definisikan data baru
x1_new = 8
x2_new = 8

# Hitung jarak antara data baru dan setiap data pada dataset
data['distance'] = data.apply(lambda row: euclidean([x1_new, x2_new], [row['X1'], row['X2']]), axis=1)

# Urutkan data berdasarkan jarak dari yang terkecil hingga yang terbesar
data_sorted = data.sort_values('distance')

# Ambil 7 data terdekat
k = 7
data_nearest = data_sorted.head(k)

# Hitung frekuensi masing-masing kelas pada K data terdekat
class_freq = data_nearest['y'].value_counts()

# Kelas dengan frekuensi terbanyak dianggap sebagai prediksi kelas dari data baru
prediction = class_freq.index[0]

print(f"Prediksi kelas untuk data baru ({x1_new}, {x2_new}) adalah {prediction}")
