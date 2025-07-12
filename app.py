import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# --- Bagian Pelatihan Model ---
try:
    # UBAH NAMA FILE CSV DI SINI
    df = pd.read_csv('student_performance.csv')
except FileNotFoundError:
    print("Error: student_performance.csv not found. Please download it and place it in the same directory.")
    exit()

# Pilih fitur yang akan digunakan berdasarkan student_performance.csv
# Fitur numerik
numerical_features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade']
# Fitur kategorikal (biner atau multi-kategori)
categorical_features = [
    'Gender',
    'ExtracurricularActivities',
    'ParentalSupport'
]
# Target
target = 'FinalGrade' # Target sekarang adalah 'FinalGrade'

# Pisahkan fitur (X) dan target (y)
X = df[numerical_features + categorical_features]
y = df[target]

# Pra-pemrosesan: Menggunakan OneHotEncoder untuk fitur kategorikal
# Passthrough untuk fitur numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Buat pipeline: Pra-pemrosesan + Model Random Forest Regressor
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(random_state=42))])

# Latih model
print("Melatih model... ini mungkin membutuhkan waktu sebentar.")
model_pipeline.fit(X, y)
print("Model selesai dilatih.")

# Simpan model yang sudah dilatih
joblib.dump(model_pipeline, 'student_performance_model.pkl')
print("Model disimpan sebagai student_performance_model.pkl")

# --- Bagian Flask App ---

@app.route('/')
def index():
    """Menampilkan halaman utama (formulir input)."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima data dari formulir, memprosesnya, dan mengembalikan prediksi."""
    data = request.json # Data akan dikirim sebagai JSON dari frontend

    # Pastikan semua fitur yang diharapkan ada dalam data
    input_data = {}
    # Sesuaikan dengan nama fitur baru
    for feature in numerical_features + categorical_features:
        input_data[feature] = data.get(feature)

    # Konversi data input ke DataFrame (penting agar sesuai dengan format model)
    input_df = pd.DataFrame([input_data])

    try:
        # Lakukan prediksi
        prediction = model_pipeline.predict(input_df)[0]
        prediction = round(prediction) # Bulatkan hasilnya
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)