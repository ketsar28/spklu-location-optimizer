# ⚡ SPKLU Location Optimizer
### Optimasi Penempatan Lokasi Stasiun Pengisian Kendaraan Listrik Umum Menggunakan Machine Learning dan Mixed-Integer Linear Programming

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spklu-location-optimizer.streamlit.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ketsar/spklu-location-optimizer)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

[Demo Langsung](https://spklu-location-optimizer.streamlit.app/) • [Dokumentasi](#-dokumentasi) • [Instalasi](#-instalasi) • [Kontak](#-kontak--sosial-media)

</div>

---

## 📖 Deskripsi Proyek

Seiring dengan akselerasi global menuju kendaraan listrik (Electric Vehicle/EV), tantangan terbesar yang dihadapi adalah pembangunan infrastruktur pengisian daya yang efisien, strategis, dan menguntungkan. Investasi pada **Stasiun Pengisian Kendaraan Listrik Umum (SPKLU)** membutuhkan modal yang sangat besar, dan penempatan yang tidak tepat dapat menyebabkan:

- 📉 Stasiun sepi pengguna (*underutilized*)
- 💰 Return on Investment (ROI) yang rendah
- 🚗 Perlambatan adopsi kendaraan listrik secara keseluruhan

**SPKLU Location Optimizer** adalah solusi berbasis data yang mengintegrasikan **Machine Learning** dan **Mathematical Optimization** untuk mengubah data mentah menjadi rekomendasi strategis yang dapat ditindaklanjuti. Proyek ini menyediakan *framework* analitis dua-tahap yang komprehensif:

1. **Tahap Prediksi**: Model Machine Learning (Gradient Boosting) memprediksi potensi permintaan di ribuan lokasi
2. **Tahap Optimisasi**: Mixed-Integer Linear Programming (MILP) menentukan alokasi optimal untuk memaksimalkan cakupan layanan

### 🎯 Tujuan Proyek

Mengubah proses penentuan lokasi SPKLU dari **spekulasi** menjadi **strategi berbasis data** yang terukur, efisien, dan menguntungkan.

---

## 🖥️ Demo Aplikasi

### 📺 Tampilan Dashboard

![Dashboard Preview](https://github.com/user-attachments/assets/8730af1e-ec2b-45fb-93b7-56e13c22f1bf)

### 🌐 Akses Langsung

- **Streamlit Cloud**: [spklu-location-optimizer.streamlit.app](https://spklu-location-optimizer.streamlit.app/)
- **Hugging Face Spaces**: [huggingface.co/spaces/ketsar/spklu-location-optimizer](https://huggingface.co/spaces/ketsar/spklu-location-optimizer)

---

## ✨ Fitur Utama

### 🗺️ **Visualisasi Interaktif**
- Peta interaktif dengan **PyDeck** menampilkan 50 lokasi optimal
- Lingkaran cakupan radius **10 km** untuk setiap lokasi
- Tooltip informatif dengan detail ZIP code, negara bagian, dan estimasi permintaan

### 🎛️ **Filter Dinamis**
- Filter berdasarkan **Negara Bagian (State)**
- Slider **Top N Locations** untuk kustomisasi tampilan
- Analisis real-time berdasarkan parameter yang dipilih

### 🧪 **Simulasi What-If Analysis**
- Input parameter hipotetis (tipe fasilitas, jaringan EV, jumlah charger, dll.)
- Prediksi permintaan instant menggunakan pre-trained model
- Visualisasi faktor-faktor kunci yang mempengaruhi prediksi

### 📊 **Model Interpretability**
- **Feature Importance Chart**: Identifikasi faktor paling berpengaruh
- Transparansi model untuk decision-making yang lebih baik

---

## 🛠️ Metodologi

### **🔬 Tahap 1: Prediksi Permintaan (Machine Learning)**

#### 1.1 Sumber Data
Data historis diperoleh dari **[U.S. Department of Energy's Alternative Fuels Data Center (AFDC)](https://afdc.energy.gov/data_download)**, mencakup:
- Lokasi stasiun pengisian existing
- Tipe fasilitas dan jaringan EV
- Jumlah charger (Level 2 dan DC Fast)
- Data geografis (ZIP code, koordinat lat/long)

#### 1.2 Feature Engineering
Data mentah diagregasi ke tingkat **ZIP Code** dengan fitur-fitur baru:
- `total_level2`: Total charger Level 2 per area
- `total_dc_fast`: Total DC Fast Charger per area
- `new_stations_last_2_years`: Momentum pertumbuhan infrastruktur
- `avg_station_age_days`: Rata-rata umur stasiun di area
- `dominant_facility_type`: Tipe fasilitas dominan
- `dominant_ev_network`: Jaringan EV dominan
- `dominant_interaction`: Kombinasi fasilitas × jaringan

#### 1.3 Model Selection & Performance
Setelah membandingkan 6 algoritma Machine Learning:
- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- **Gradient Boosting** ✅ (Terpilih)
- XGBoost

**Hasil Terbaik**: **Gradient Boosting**
- **R-squared**: **0.91+**
- **Mean Absolute Error (MAE)**: Rendah
- **Stabilitas**: Konsisten pada cross-validation

#### 1.4 Model Interpretability
Menggunakan **SHAP (SHapley Additive exPlanations)** untuk:
- Membongkar "black box" model
- Memastikan keputusan model masuk akal secara bisnis
- Meningkatkan kepercayaan stakeholder

---

### **🎯 Tahap 2: Optimisasi Alokasi (Mixed-Integer Linear Programming)**

#### 2.1 Formulasi Masalah
Masalah ini diformulasikan sebagai **Facility Location Problem** dengan MILP:

**Variabel Keputusan:**
- Binary variable: `build[i]` ∈ {0, 1} untuk setiap lokasi kandidat

**Objective Function:**
```
Maximize: Σ (predicted_demand[i] × build[i])
```

**Constraints:**
1. Budget Constraint: `Σ build[i] ≤ 50` (maksimal 50 SPKLU baru)
2. Coverage Constraint: Setiap SPKLU melayani radius 10 km
3. Binary Constraint: `build[i] ∈ {0, 1}`

#### 2.2 Tools & Solver
- **Framework**: **Pyomo** (Python Optimization Modeling Objects)
- **Solver**: **CBC** (COIN-OR Branch and Cut) - open-source solver
- **Computational Strategy**: Optimisasi pada **top 1000 locations** (berdasarkan predicted demand)

#### 2.3 Output
50 lokasi optimal dengan:
- Koordinat geografis (latitude, longitude)
- Estimasi permintaan yang dapat dilayani
- Negara bagian (State)
- ZIP code

---

## 💻 Tech Stack

### **Backend & Data Processing**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### **Machine Learning & Optimization**
- **Scikit-learn**: Model training & evaluation
- **XGBoost & LightGBM**: Advanced gradient boosting
- **Pyomo**: Optimization modeling framework
- **CBC Solver**: Mixed-integer linear programming solver
- **SHAP**: Model interpretability
- **Feature-engine**: Feature engineering pipeline
- **Geopy**: Geocoding & distance calculations

### **Visualization & Dashboard**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)

- **Streamlit**: Interactive web application
- **PyDeck**: Advanced geospatial visualization
- **Plotly Express**: Interactive charts

### **Deployment**
- **Streamlit Community Cloud**: Cloud hosting
- **Hugging Face Spaces**: Alternative deployment platform
- **GitHub**: Version control & collaboration

---

## 🚀 Instalasi

### **Prerequisites**
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Git

### **Langkah Instalasi**

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/ketsar28/spklu-location-optimizer.git
cd spklu-location-optimizer
```

#### 2️⃣ Buat Virtual Environment (Opsional tapi Disarankan)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Jalankan Aplikasi
```bash
streamlit run spklu_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

---

## 📊 Struktur Project

```
spklu-location-optimizer/
│
├── spklu_app.py                          # Main Streamlit application
├── best_model_Gradient_Boosting.pkl     # Pre-trained ML model
├── rekomendasi_lokasi_spklu.csv         # Optimization results (50 optimal locations)
├── zip_to_state_geodata.csv             # ZIP code to State mapping
├── ev_analysis.ipynb                     # Jupyter notebook for EDA & modeling
│
├── requirements.txt                      # Python dependencies
├── Procfile                             # Deployment configuration
├── setup.sh                             # Setup script for deployment
├── Dockerfile.txt                       # Docker configuration (if needed)
│
└── README.md                            # Project documentation (this file)
```

---

## 📚 Dokumentasi

### **Penggunaan Dashboard**

#### Tab 1: 📌 Rekomendasi Lokasi Optimal
1. Gunakan filter **Negara Bagian** di sidebar untuk fokus pada region tertentu
2. Atur slider **Top N Locations** untuk menampilkan jumlah lokasi yang diinginkan
3. Hover pada peta untuk melihat detail setiap lokasi
4. Scroll ke bawah untuk melihat tabel data lengkap

#### Tab 2: 🔬 Simulasi & Insight Model
1. Masukkan parameter di sidebar:
   - Tipe Fasilitas (Hotel, Gas Station, dll.)
   - Jaringan EV (ChargePoint, Tesla, dll.)
   - Jumlah Charger (Level 2 dan DC Fast)
   - Rata-rata umur stasiun (dalam hari)
2. Klik tombol **"Prediksi"** untuk mendapatkan estimasi permintaan
3. Lihat chart **Feature Importance** untuk memahami faktor kunci

#### Tab 3: 📄 Informasi Proyek
- Penjelasan lengkap metodologi dua-tahap
- Dokumentasi teknis dan sumber data

---

## 📈 Hasil & Insights

### **Key Findings**
1. **Faktor Terpenting** dalam memprediksi permintaan:
   - `total_level2`: Kepadatan infrastruktur existing (importance: ~35%)
   - `new_stations_last_2_years`: Momentum pertumbuhan area (importance: ~25%)
   - Kombinasi Jaringan & Fasilitas (importance: ~15%)

2. **Lokasi Optimal** terkonsentrasi di:
   - Area urban dengan kepadatan populasi tinggi
   - Koridor transportasi utama (interstate highways)
   - Negara bagian dengan regulasi pro-EV (California, Washington, dll.)

3. **ROI Potensial**: Lokasi yang direkomendasikan memiliki estimasi permintaan 3-5x lebih tinggi dibanding lokasi random

---

## 🔧 Development & Contribution

### **Roadmap Future Enhancements**
- [ ] Integrasi dengan real-time traffic data
- [ ] Model reinforcement learning untuk dynamic pricing
- [ ] Multi-objective optimization (cost, demand, environmental impact)
- [ ] Mobile app version
- [ ] API endpoint untuk integrasi dengan sistem lain

### **Contributing**
Kontribusi sangat diterima! Silakan:
1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan Anda (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

---

## 📄 Data Sources

### **Primary Dataset**
**Alternative Fueling Station Locator**
- **Provider**: U.S. Department of Energy's Alternative Fuels Data Center (AFDC)
- **Link**: [afdc.energy.gov/data_download](https://afdc.energy.gov/data_download)
- **Coverage**: 50,000+ EV charging stations di Amerika Serikat
- **Update Frequency**: Monthly

### **Auxiliary Data**
- ZIP Code to State mapping: U.S. Census Bureau
- Geospatial data: OpenStreetMap via Geopy

---

## 🎓 Academic References

Proyek ini terinspirasi dari penelitian terkait:
- Facility Location Problems (FLP) in Operations Research
- Maximum Coverage Location Problem (MCLP)
- Machine Learning for Demand Forecasting
- Sustainable Transportation Infrastructure Planning

---

## 👤 Author

<div align="center">

### **Muhammad Ketsar Ali Abi Wahid**

*Data Scientist | Machine Learning Engineer | Optimization Enthusiast*

</div>

---

## 📱 Kontak & Sosial Media

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-ketsar28-181717?style=for-the-badge&logo=github)](https://github.com/ketsar28/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ketsarali-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ketsarali/)
[![Instagram](https://img.shields.io/badge/Instagram-ketsar.aaw-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/ketsar.aaw/)

[![HuggingFace](https://img.shields.io/badge/HuggingFace-ketsar-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ketsar)
[![Streamlit](https://img.shields.io/badge/Streamlit-ketsar28-FF4B4B?style=for-the-badge&logo=streamlit)](https://share.streamlit.io/user/ketsar28)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-Contact_Me-25D366?style=for-the-badge&logo=whatsapp)](https://api.whatsapp.com/send/?phone=6285155343380&text=Hi%20Ketsar,%20saya%20tertarik%20dengan%20project%20SPKLU%20Location%20Optimizer!)

</div>

---

## 📝 License & Copyright

<div align="center">

**Copyright © 2025 Muhammad Ketsar Ali Abi Wahid**

All rights reserved.

This project and all associated code, documentation, and materials are the intellectual property of **Muhammad Ketsar Ali Abi Wahid**.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</div>

---

<div align="center">

**⭐ Jika project ini bermanfaat, jangan lupa berikan star di GitHub! ⭐**

*Made with ❤️ and ☕ by Muhammad Ketsar Ali Abi Wahid*

**[↑ Kembali ke Atas](#-spklu-location-optimizer)**

</div>
