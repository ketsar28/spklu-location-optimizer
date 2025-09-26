# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:42:17 2025

@author: KETSAR
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sklearn
from sklearn import set_config
import joblib
import pydeck as pdk
set_config(transform_output='pandas')

st.set_page_config(
        page_title='Dashboard Optimasi Lokasi SPKLU',
        page_icon='⚡',
        layout='wide',
)

@st.cache_data
def load_recommendation_data(path):
    try:
        df = pd.read_csv(path)
        
        required_cols = {'id', 'avg_latitude', 'avg_longitude', 'demand'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"Gagal memuat data. Kolom berikut tidak ditemukan di CSV: {missing}", icon="🚨")
            return None
        
        df['id'] = df['id'].astype(str)
        
        df['avg_latitude'] = pd.to_numeric(df['avg_latitude'])
        df['avg_longitude'] = pd.to_numeric(df['avg_longitude'])
        df.rename(columns={
                'id':'zip_code',
                'demand':'predicted_demand_covered'
            }, inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {path}. Pastikan nama file CSV sudah benar.", icon="🚨")
        return None
    except KeyError as e :
        st.error(f"Gagal memuat data. Kolom {e} tidak ditemukan di CSV.", icon="🚨")
        return None
    
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model tidak ditemukan: {path}. Pastikan 'best_model_Gradient_Boosting.pkl' yang baru ada.", icon='🚨')
        return None

    
recommend_data = load_recommendation_data('rekomendasi_lokasi_spklu.csv')
model = load_model('best_model_Gradient_Boosting.pkl')

try:
    # df_full = pd.read_csv('https://drive.google.com/uc?export=download&id=1x3TYeqXpDxGdKmbbhhtH-m8-kBcUgF8i', usecols=['ZIP', 'State'], dtype={'ZIP': str})
    # df_states = df_full.drop_duplicates('ZIP').rename(columns={'ZIP': 'zip_code'})
    df_states_map = pd.read_csv('zip_to_state_geodata.csv', dtype={'ZIP': str})
    df_states_map.rename(columns={'ZIP': 'zip_code'}, inplace=True)
    if recommend_data is not None:
        recommend_data = pd.merge(recommend_data, df_states_map, on='zip_code', how='left')
except FileNotFoundError:
    st.warning("File data asli tidak ditemukan, filter State tidak akan tersedia.")


with st.sidebar:
    st.image('https://cdn.motor1.com/images/mgl/g44JxN/s3/polestar-2-at-a-shell-fast-charging-station-abb-chargers.jpg', use_container_width=True)
    st.title('Panel Kontrol')
    st.divider()
    st.subheader('⚙️ Masukan Data Untuk Prediksi')
    st.caption('Masukkan data untuk lokasi hipotetis untuk mendapatkan estimasi jumlah stasiun (permintaan).')
    
    facility_type = ['HOTEL', 'CAR_DEALER', 'PUBLIC', 'OFFICE_BLDG', 'PARKING_LOT', 'FED_GOV', 
                     'GAS_STATION', 'MUNI_GOV', 'SHOPPING_CENTER', 'RESTAURANT', 'COLLEGE_CAMPUS', 'CONVENIENCE_STORE', 'OTHER']
    ev_network_type = ['ChargePoint Network', 'Non-Networked', 'Blink Network', 'Tesla Destination',
                  'Tesla', 'EV Connect', 'AMPUP', 'SHELL_RECHARGE', 'FLO', 'Electrify America', 'EVgo', 'OTHER_NETWORK']
    
    facility_input = st.selectbox('Tipe Fasilitas :', facility_type)
    ev_network_input = st.selectbox('Jaringan EV :', ev_network_type)
    level2_charger_input = st.slider('Jumlah Charger Level 2 :', 0, 1056, 20)
    dc_charger_input = st.slider('Jumlah DC Fast Charger :', 0, 151, 2)
    station_age_days_input = st.slider('Rata-rata Umur Stasiun (hari) :', 0, 8519, 300, help='Perkiraan Rata-Rata Umur Stasiun (hari) Di Area Tersebut')
    
    days = station_age_days_input
    years = days // 365
    months = (days % 365) // 30
    remaining_days = (days % 365) % 30
    
    st.caption(f'{days} Hari = {years} Tahun {months} Bulan {remaining_days} Hari')
    st.divider()
    
    predict_button = st.button(type='primary', label='Prediksi', use_container_width=True)
    
    
st.title('⚡ Dashboard Optimasi Penempatan SPKLU')
st.caption('Alat bantu pengambilan keputusan berbasis data untuk investasi infrastruktur kendaraan listrik.')
tab1, tab2, tab3 = st.tabs(['📌 Rekomendasi Lokasi Optimal', '🔬 Simulasi & Insight Model', '📄 Informasi Proyek'])


with tab1:
    st.header('Maps Sebaran Lokasi Hasil Optimisasi')
    if recommend_data is not None:
        
        st.sidebar.divider()
        st.sidebar.subheader("Filter Peta Rekomendasi")
        
        display_data = recommend_data.copy()
        
        # filter berdasarkan State
        if 'State' in display_data.columns:
            all_states = display_data['State'].dropna().unique().tolist()
            all_states.sort()
            
            default_states = recommend_data['State'].value_counts().head(5).index.tolist()
            selected_states = st.sidebar.multiselect("Pilih Negara Bagian (State):", all_states, default=default_states)
            if selected_states:
                display_data = recommend_data[recommend_data['State'].isin(selected_states)]
        
        if not display_data.empty:
            min_val = 1 
            max_val = len(display_data)
            
            if max_val > min_val:
                default_val = min(50, max_val)
                top_n = st.sidebar.slider("Tampilkan Top N Lokasi:", min_val, max_val, default_val)
                display_data = display_data.sort_values('predicted_demand_covered', ascending=False).head(top_n)
            else:
                top_n = 1
        
        st.sidebar.divider()
        st.sidebar.markdown(
                """
                <div style="text-align: center;">
                    <small>
                        © 2025 - Muhammad Ketsar Ali Abi Wahid <br>
                        Final Project - Data Science Bootcamp <br>
                        PT Epam Digital Mandiri
                    </small>
                </div>
                """,
                unsafe_allow_html=True
         )
        
        m1, m2, m3 = st.columns(3)
        m1.metric('Total Rekomendasi Lokasi :', display_data.shape[0])
        m2.metric('Total Estimasi Permintaan Tercover', display_data.predicted_demand_covered.sum())
        
        mean_demand = display_data['predicted_demand_covered'].mean() if not display_data.empty else 0
        m3.metric("Rata-rata Permintaan per Lokasi", f"{mean_demand:.2f}")
        st.divider()
        
        if not display_data.empty:
            display_data['lat_display'] = display_data['avg_latitude'].map('{:.2f}'.format)
            display_data['lon_display'] = display_data['avg_longitude'].map('{:.2f}'.format)
            
            view_state = pdk.ViewState(
                longitude=display_data['avg_longitude'].mean(), 
                latitude=display_data['avg_latitude'].mean(), 
                zoom=3.5, 
                pitch=50
            )
            
            coverage_layer = pdk.Layer(
                'ScatterplotLayer', 
                data=display_data, 
                get_position='[avg_longitude, avg_latitude]', 
                get_color='[255, 30, 30, 50]',
                get_radius=10000, 
                pickable=False,
                auto_highlight=True
            )
            
            station_layer = pdk.Layer(
                'ScatterplotLayer',
                data=display_data,
                get_position='[avg_longitude, avg_latitude]',
                get_color='[255, 30, 30, 200]', 
                get_radius=2000, 
                pickable=True,
                auto_highlight=True
            )
            
            tooltip = {
                "html": "Kode Pos : <b>{zip_code}</b></br>"
                        "Negara Bagian : <b>{State}</b></br>"
                        "Estimasi Permintaan : <b>{predicted_demand_covered}</b></br>"
                        "Lintang (latitude) : <b>{lat_display}</b></br>"
                        "Bujur (longitude) : <b>{lon_display}</b>"
            }
            
            peta_rekomendasi = pdk.Deck(
                layers=[coverage_layer, station_layer], 
                initial_view_state=view_state, 
                map_style='mapbox://styles/mapbox/dark-v10', 
                tooltip=tooltip
            )
            
            st.pydeck_chart(peta_rekomendasi, use_container_width=True)
            
            st.subheader('Detail Data Lokasi Rekomendasi')
            st.dataframe(display_data.drop(columns=['lat_display', 'lon_display']), use_container_width=True)
    else:
        st.error('Data rekomendasi tidak dapat dimuat.')
        

with tab2:
    st.header('Hasil Simulasi & Faktor Kunci')
    col1, col2 = st.columns(2, gap='large')
    
    with col1:
        st.subheader('Hasil Prediksi Permintaan')
        if predict_button:
            if model:
                input_data={
                        "total_level2" : level2_charger_input,
                        "total_dc_fast" : dc_charger_input,
                        "dominant_facility_type" : facility_input,
                        "dominant_ev_network" : ev_network_input,
                        "avg_station_age_days" : station_age_days_input, 
                        "dominant_interaction" : f'{facility_input}_{ev_network_input}',
                        "new_station_last_2_years" : 2,
                        "last_station_opened_days_ago" : station_age_days_input * 0.5
                    }
                df_user_input = pd.DataFrame(input_data, index=[0])
                
                try:
                    prediction = model.predict(df_user_input)
                    predict_demand = round(prediction[0], 0)
                    
                    st.metric(
                        label="Estimasi Jumlah Stasiun (Demand)",
                        value=f"~ {predict_demand}",
                        help='Nilai ini merepresentasikan potensi permintaan. Semakin tinggi, semakin baik.'
                    )
                    with st.expander('Lihat detail input mentah yang dikirim ke model'):
                        st.dataframe(input_data)
                except Exception as e : 
                    st.error(f"Terjadi error saat prediksi: {e}", icon="🚨")
            else:
                st.error("Model prediksi tidak dapat dimuat.")
        else:
            st.error("Hasil prediksi akan muncul di sini setelah menekan tombol 'Prediksi' di sidebar.")

    with col2:
        st.subheader("Faktor Paling Berpengaruh")
        if model:
            try:
                regressor = model.named_steps.get('regressor') or model.named_steps.get('gb')
                feature_names = regressor.feature_names_in_
                importances = regressor.feature_importances_
                
                df_features = pd.DataFrame({'Feature': feature_names, 'Importances': importances}).sort_values(by='Importances', ascending=False).head(10)
                
                fig = px.bar(df_features, x='Importances', y='Feature', orientation='h', template='plotly_dark', color_discrete_sequence=['#ff4b4b'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Tingkat Pengaruh", yaxis_title="Faktor")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e :
                st.error(f"Terjadi error saat prediksi : {e}", icon='🚨')
     

with tab3:
    st.header('📄 Tentang Proyek Optimisasi Penempatan SPKLU')
    st.image('https://futuretransport-news.com/wp-content/uploads/sites/3/2021/12/Tritium-Shell.png', 
             use_container_width=True)

    st.markdown("""
    Proyek ini adalah sebuah *proof-of-concept* yang menunjukkan bagaimana **analisis data, machine learning, dan optimisasi matematis** dapat digabungkan untuk menyelesaikan masalah bisnis nyata: di mana lokasi terbaik untuk membangun SPKLU baru? Tujuannya adalah untuk menggantikan pengambilan keputusan berbasis intuisi dengan rekomendasi strategis yang didukung oleh data.
    """)
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🔬 Tahap 1: Prediksi Permintaan (Machine Learning)")
        st.markdown("""
        Langkah pertama adalah memahami 'DNA' dari area-area yang sudah memiliki banyak SPKLU.
        - **Model Terbaik:** Setelah membandingkan 6 algoritma, **Gradient Boosting** terpilih karena memiliki performa paling akurat dan stabil (R-squared: **0.91**).
        - **Faktor Kunci:** Model menemukan bahwa faktor terpenting untuk memprediksi permintaan adalah:
            1.  **`total_level2`** (Kepadatan infrastruktur yang ada)
            2.  **`new_stations_last_2_years`** (Momentum pertumbuhan area)
            3.  **Kombinasi Jaringan & Fasilitas** (misal: Jaringan ChargePoint)
        - **Sumber Data:** Analisis ini menggunakan data publik dari **U.S. Alternative Fuels Data Center (AFDC)**.
        """)

    with col2:
        st.subheader("🎯 Tahap 2: Optimisasi Alokasi (MILP)")
        st.markdown("""
        Setelah mendapatkan prediksi permintaan untuk setiap area, langkah selanjutnya adalah menentukan 50 lokasi terbaik dari ribuan kandidat.
        - **Metodologi:** **Mixed-Integer Linear Programming (MILP)** digunakan untuk menyelesaikan masalah ini, karena melibatkan keputusan biner ('bangun' atau 'tidak').
        - **Tools:** Model optimisasi ini diformulasikan menggunakan *framework* **Pyomo** dan diselesaikan dengan *solver* open-source **CBC**.
        - **Tujuan (Objective):**
            - **Memaksimalkan** total prediksi permintaan yang dapat dilayani.
        - **Batasan (Constraints):**
            1.  Hanya membangun **maksimal 50 SPKLU baru**.
            2.  Setiap SPKLU dianggap melayani area dalam **radius 10 km**.
        """)

    st.divider()
    st.caption("Dashboard ini memvisualisasikan hasil dari kedua tahapan tersebut, bertujuan untuk membantu pemangku kepentingan membuat keputusan investasi yang lebih cerdas.")
        
        
        
        
        
        
        
        
        
        
        
    