# ⚡ Optimasi Penempatan Lokasi SPKLU Menggunakan Machine Learning dan MILP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spklu-location-optimizer.streamlit.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ketsar/spklu-location-optimizer)

Seiring dengan transisi global menuju kendaraan listrik (EV), tantangan utama yang muncul adalah pembangunan infrastruktur pengisian daya yang efisien dan strategis. Investasi pada Stasiun Pengisian Kendaraan Listrik Umum (SPKLU) melibatkan modal yang sangat besar, dan penempatan yang tidak tepat dapat menyebabkan stasiun sepi pengguna (underutilized), return on investment (ROI) yang rendah, dan pada akhirnya memperlambat laju adopsi EV secara keseluruhan. Keputusan yang hanya berdasarkan intuisi atau ketersediaan lahan tidak lagi memadai untuk mengatasi masalah berskala besar ini.

Repositori ini menyajikan sebuah kerangka kerja analitis yang komprehensif untuk menjawab tantangan tersebut. Proyek ini mengimplementasikan pendekatan hibrida dua-tahap yang mengubah data mentah menjadi rekomendasi strategis yang dapat ditindaklanjuti. Tahap pertama melibatkan pengembangan model machine learning (Gradient Boosting) yang dilatih untuk memprediksi potensi permintaan di ribuan lokasi. Tahap kedua menggunakan hasil prediksi tersebut sebagai input untuk model Mixed-Integer Linear Programming (MILP) yang secara matematis menentukan alokasi optimal dari sejumlah stasiun baru untuk memaksimalkan cakupan layanan di seluruh jaringan.

Tujuan utama proyek ini adalah untuk menyediakan sebuah proof-of-concept dan alat bantu keputusan yang kuat, mengubah proses penentuan lokasi dari spekulasi menjadi strategi berbasis data yang terukur dan efisien.

###  Demo Aplikasi Langsung

* **Streamlit Cloud:** [https://spklu-location-optimizer.streamlit.app/](https://spklu-location-optimizer.streamlit.app/)
* **Hugging Face Spaces:** [https://huggingface.co/spaces/ketsar/spklu-location-optimizer](https://huggingface.co/spaces/ketsar/spklu-location-optimizer)

### 🖥️ Tampilan Dashboard

![Dashboard Preview](https://github.com/user-attachments/assets/8730af1e-ec2b-45fb-93b7-56e13c22f1bf)
---

### 📋 Ringkasan Proyek

Penempatan SPKLU yang tidak efisien merupakan masalah investasi berisiko tinggi yang dapat menghambat adopsi kendaraan listrik. Proyek ini mengatasi tantangan tersebut dengan menggabungkan kekuatan **prediksi machine learning** untuk memahami potensi permintaan dan **optimisasi matematis** untuk alokasi sumber daya yang terbatas. Hasil akhirnya adalah sebuah *dashboard* interaktif yang memvisualisasikan **50 lokasi paling optimal** untuk pembangunan SPKLU baru di Amerika Serikat, berdasarkan data historis.

### ✨ Fitur Utama

* **Peta Interaktif:** Visualisasi lokasi SPKLU yang direkomendasikan lengkap dengan lingkaran cakupan radius 10 km.
* **Filter Dinamis:** Pengguna dapat memfilter rekomendasi berdasarkan Negara Bagian (State) dan jumlah lokasi teratas (Top N).
* **Simulasi Permintaan:** Fitur *what-if analysis* untuk memprediksi potensi permintaan di lokasi hipotetis.
* **Interpretasi Model:** Menampilkan faktor-faktor kunci (*feature importance* dan SHAP plot) yang paling mempengaruhi prediksi model.

---

### 🛠️ Metodologi

Proyek ini menggunakan pendekatan dua-tahap yang sistematis:

#### **Tahap 1: Prediksi Permintaan (Machine Learning)**
1.  **Sumber Data:** Data historis diolah dari **[U.S. Department of Energy's Alternative Fuels Data Center (AFDC)](https://afdc.energy.gov/data_download)**.
2.  **Feature Engineering:** Data mentah diagregasi ke tingkat ZIP Code dan fitur-fitur baru diciptakan untuk menangkap karakteristik area, seperti kepadatan charger dan momentum pertumbuhan.
3.  **Pemilihan Model:** Setelah membandingkan 6 algoritma, **Gradient Boosting** terpilih sebagai yang paling akurat dengan **R-squared > 0.91**.
4.  **Interpretasi:** SHAP (SHapley Additive exPlanations) digunakan untuk "membongkar" logika model dan memastikan model membuat keputusan yang masuk akal.

#### **Tahap 2: Optimisasi Alokasi (Mixed-Integer Linear Programming)**
1.  **Metodologi:** Masalah ini diformulasikan sebagai *Mixed-Integer Linear Programming* (MILP) untuk memaksimalkan cakupan layanan.
2.  **Tools:** Menggunakan *framework* **Pyomo** dan *solver* open-source **CBC**.
3.  **Tujuan & Batasan:**
    * **Tujuan:** Memaksimalkan total prediksi permintaan yang dapat dilayani.
    * **Batasan:**
        * Membangun maksimal **50 SPKLU baru**.
        * Setiap SPKLU dianggap melayani area dalam **radius 10 km**.
4.  **Strategi Komputasi:** Analisis dilakukan pada **sampel 1000 lokasi teratas** berdasarkan potensi permintaan untuk menjaga agar proses optimisasi berjalan cepat.

---

### 💻 Tumpukan Teknologi (Tech Stack)

* **Analisis & Pemodelan:** Python, Pandas, Scikit-learn, Geopy, Joblib
* **Optimisasi:** Pyomo, CBC Solver
* **Visualisasi & Dashboard:** Streamlit, Pydeck, Plotly Express
* **Deployment:** GitHub, Streamlit Community Cloud, Hugging Face Spaces

---

### 🚀 Cara Menjalankan Proyek Secara Lokal

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/ketsar28/spklu-location-optimizer.git](https://github.com/ketsar28/spklu-location-optimizer.git)
    cd spklu-location-optimizer
    ```

2.  **(Opsional) Buat dan aktifkan virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **Instal semua dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run spklu_app.py
    ```

---

### 📄 Sumber Data

* Dataset utama yang digunakan dalam proyek ini adalah **Alternative Fueling Station Locator** yang disediakan oleh **[U.S. Department of Energy's Alternative Fuels Data Center (AFDC)](https://afdc.energy.gov/data_download)**.

---

### 👤 Author

* **Muhammad Ketsar Ali Abi Wahid**
* **LinkedIn:** [linkedin.com/in/ketsarali](https://www.linkedin.com/in/ketsarali/)
* **Instagram:** [@ketsar.aaw](https://www.instagram.com/ketsar.aaw/)
* **GitHub:** [@ketsar28](https://github.com/ketsar28)
