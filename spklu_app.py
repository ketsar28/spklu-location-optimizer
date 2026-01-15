# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:42:17 2025

@author: KETSAR
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn import set_config
import joblib
import pydeck as pdk
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import json
set_config(transform_output='pandas')


st.set_page_config(
    page_title='Dashboard Optimasi Lokasi SPKLU',
    page_icon='⚡',
    layout='wide',
)

# Setup Mapbox API Token (untuk style peta premium)
# Token diambil dari .streamlit/secrets.toml
if "MAPBOX_TOKEN" in st.secrets:
    os.environ["MAPBOX_API_KEY"] = st.secrets["MAPBOX_TOKEN"]
    map_style_config = 'mapbox://styles/mapbox/dark-v11'
else:
    # Fallback ke style Open-Source (CartoDB) kalau token tidak ada
    map_style_config = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'


@st.cache_data(ttl=660)
def load_data_from_bq():
    gcp_service_account = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        gcp_service_account)

    client = bigquery.Client(credentials=credentials,
                             project=credentials.project_id)

    query = """
    select 
        station_name,latitude,longitude,city, state, fuel_type, status
    from `personal-480906.raw_spklu_data.clean_fuel_stations`
    where status = 'E'
    """

    df = client.query(query).to_dataframe()
    return df


@st.cache_data
def load_recommendation_data(path):
    try:
        df = pd.read_csv(path)

        required_cols = {'id', 'avg_latitude', 'avg_longitude', 'demand'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(
                f"Gagal memuat data. Kolom berikut tidak ditemukan di CSV: {missing}", icon="🚨")
            return None

        df['id'] = df['id'].astype(str)

        df['avg_latitude'] = pd.to_numeric(df['avg_latitude'])
        df['avg_longitude'] = pd.to_numeric(df['avg_longitude'])
        df.rename(columns={
            'id': 'zip_code',
            'demand': 'predicted_demand_covered'
        }, inplace=True)
        return df
    except FileNotFoundError:
        st.error(
            f"File tidak ditemukan: {path}. Pastikan nama file CSV sudah benar.", icon="🚨")
        return None
    except KeyError as e:
        st.error(
            f"Gagal memuat data. Kolom {e} tidak ditemukan di CSV.", icon="🚨")
        return None


@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(
            f"Model tidak ditemukan: {path}. Pastikan 'best_model_Gradient_Boosting.pkl' yang baru ada.", icon='🚨')
        return None


recommend_data = load_recommendation_data('rekomendasi_lokasi_spklu.csv')
df_data_asli = load_data_from_bq()
model = load_model('best_model_Gradient_Boosting.pkl')

try:
    # df_full = pd.read_csv('https://drive.google.com/uc?export=download&id=1x3TYeqXpDxGdKmbbhhtH-m8-kBcUgF8i', usecols=['ZIP', 'State'], dtype={'ZIP': str})
    # df_states = df_full.drop_duplicates('ZIP').rename(columns={'ZIP': 'zip_code'})
    df_states_map = pd.read_csv('zip_to_state_geodata.csv', dtype={'ZIP': str})
    df_states_map.rename(columns={'ZIP': 'zip_code'}, inplace=True)
    if recommend_data is not None:
        recommend_data = pd.merge(
            recommend_data, df_states_map, on='zip_code', how='left')
except FileNotFoundError:
    st.warning("File data asli tidak ditemukan, filter State tidak akan tersedia.")


with st.sidebar:
    st.image('https://cdn.motor1.com/images/mgl/g44JxN/s3/polestar-2-at-a-shell-fast-charging-station-abb-chargers.jpg',
             use_container_width=True)
    st.title('Panel Kontrol')
    st.divider()
    st.subheader('⚙️ Masukan Data Untuk Prediksi')
    st.caption(
        'Masukkan data untuk lokasi hipotetis untuk mendapatkan estimasi jumlah stasiun (permintaan).')

    facility_type = ['HOTEL', 'CAR_DEALER', 'PUBLIC', 'OFFICE_BLDG', 'PARKING_LOT', 'FED_GOV',
                     'GAS_STATION', 'MUNI_GOV', 'SHOPPING_CENTER', 'RESTAURANT', 'COLLEGE_CAMPUS', 'CONVENIENCE_STORE', 'OTHER']
    ev_network_type = ['ChargePoint Network', 'Non-Networked', 'Blink Network', 'Tesla Destination',
                       'Tesla', 'EV Connect', 'AMPUP', 'SHELL_RECHARGE', 'FLO', 'Electrify America', 'EVgo', 'OTHER_NETWORK']

    facility_input = st.selectbox('Tipe Fasilitas :', facility_type)
    ev_network_input = st.selectbox('Jaringan EV :', ev_network_type)
    level2_charger_input = st.slider('Jumlah Charger Level 2 :', 0, 1056, 20)
    dc_charger_input = st.slider('Jumlah DC Fast Charger :', 0, 151, 2)
    station_age_days_input = st.slider('Rata-rata Umur Stasiun (hari) :', 0,
                                       8519, 300, help='Perkiraan Rata-Rata Umur Stasiun (hari) Di Area Tersebut')

    days = station_age_days_input
    years = days // 365
    months = (days % 365) // 30
    remaining_days = (days % 365) % 30

    st.caption(
        f'{days} Hari = {years} Tahun {months} Bulan {remaining_days} Hari')
    st.divider()

    predict_button = st.button(
        type='primary', label='Prediksi', use_container_width=True)


st.title('⚡ Dashboard Optimasi Penempatan SPKLU')
st.caption('Alat bantu pengambilan keputusan berbasis data untuk investasi infrastruktur kendaraan listrik.')
tab1, tab2, tab3, tab4 = st.tabs(
    ['📌 Rekomendasi Lokasi Optimal', '🔬 Simulasi & Insight Model', '📄 Informasi Proyek', '📊  Monitoring Real-Time'])


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

            default_states = recommend_data['State'].value_counts().head(
                5).index.tolist()
            selected_states = st.sidebar.multiselect(
                "Pilih Negara Bagian (State):", all_states, default=default_states)
            if selected_states:
                display_data = recommend_data[recommend_data['State'].isin(
                    selected_states)]

        if not display_data.empty:
            min_val = 1
            max_val = len(display_data)

            if max_val > min_val:
                default_val = min(50, max_val)
                top_n = st.sidebar.slider(
                    "Tampilkan Top N Lokasi:", min_val, max_val, default_val)
                display_data = display_data.sort_values(
                    'predicted_demand_covered', ascending=False).head(top_n)
            else:
                top_n = 1

        st.sidebar.divider()
        st.sidebar.markdown(
            """
                <div style="text-align: center;">
                    <small>
                        © 2025 - Muhammad Ketsar Ali Abi Wahid <br>
                        PT Epam Digital Mandiri
                    </small>
                </div>
                """,
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        m1.metric('Total Rekomendasi Lokasi :', display_data.shape[0])
        m2.metric('Total Estimasi Permintaan Tercover',
                  display_data.predicted_demand_covered.sum())

        mean_demand = display_data['predicted_demand_covered'].mean(
        ) if not display_data.empty else 0
        m3.metric("Rata-rata Permintaan per Lokasi", f"{mean_demand:.2f}")
        st.divider()

        if not display_data.empty:
            display_data['lat_display'] = display_data['avg_latitude'].map(
                '{:.2f}'.format)
            display_data['lon_display'] = display_data['avg_longitude'].map(
                '{:.2f}'.format)

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
                map_style=map_style_config,  # Pakai variabel safe config
                tooltip=tooltip
            )

            st.pydeck_chart(peta_rekomendasi, use_container_width=True)

            st.subheader('Detail Data Lokasi Rekomendasi')
            st.dataframe(display_data.drop(
                columns=['lat_display', 'lon_display']), use_container_width=True)
            st.dataframe(df_data_asli, use_container_width=True)
    else:
        st.error('Data rekomendasi tidak dapat dimuat.')


with tab2:
    st.header('Hasil Simulasi & Faktor Kunci')
    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.subheader('Hasil Prediksi Permintaan')
        if predict_button:
            if model:
                input_data = {
                    "total_level2": level2_charger_input,
                    "total_dc_fast": dc_charger_input,
                    "dominant_facility_type": facility_input,
                    "dominant_ev_network": ev_network_input,
                    "avg_station_age_days": station_age_days_input,
                    "dominant_interaction": f'{facility_input}_{ev_network_input}',
                    "new_station_last_2_years": 2,
                    "last_station_opened_days_ago": station_age_days_input * 0.5
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
                except Exception as e:
                    st.error(f"Terjadi error saat prediksi: {e}", icon="🚨")
            else:
                st.error("Model prediksi tidak dapat dimuat.")
        else:
            st.error(
                "Hasil prediksi akan muncul di sini setelah menekan tombol 'Prediksi' di sidebar.")

    with col2:
        st.subheader("Faktor Paling Berpengaruh")
        if model:
            try:
                regressor = model.named_steps.get(
                    'regressor') or model.named_steps.get('gb')
                feature_names = regressor.feature_names_in_
                importances = regressor.feature_importances_

                df_features = pd.DataFrame({'Feature': feature_names, 'Importances': importances}).sort_values(
                    by='Importances', ascending=False).head(10)

                fig = px.bar(df_features, x='Importances', y='Feature', orientation='h',
                             template='plotly_dark', color_discrete_sequence=['#ff4b4b'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  xaxis_title="Tingkat Pengaruh", yaxis_title="Faktor")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
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

with tab4:
    st.header('📊  Real-time SPKLU Monitoring')
    st.caption(
        f'Data langsung dari Google BigQuery: `personal-480906.raw_spklu_data.clean_fuel_stations`')

    # kpi scorecards (important number)
    col1, col2, col3 = st.columns(3)
    total_stations = len(df_data_asli)
    top_fuel = df_data_asli['fuel_type'].mode()[0]
    total_state = df_data_asli['state'].nunique()

    col1.metric('Total Stasiun Aktif', f'{total_stations:,}')
    col2.metric('Bahan Bakar Terpopuler', top_fuel)
    col3.metric('Cakupan Negara Bagian', f'{total_state} Bagian')

    st.divider()
    # spread maps (with scatterplot map)
    st.subheader('🗺️ Peta Sebaran Stasiun (Live)')
    # using pydeck for making maps interactive & pretty
    map_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_data_asli,
        get_position='[longitude, latitude]',
        get_color='[0,128,255,160]',
        get_radius=5000,
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=df_data_asli['latitude'].mean(),
        longitude=df_data_asli['longitude'].mean(),
        zoom=3,
        pitch=0
    )

    tooltip_live = {
        "html": "<b>{station_name}</b><br/>"
                "Kota: {city}, {state}<br/>"
                "Tipe: {fuel_type}"
    }

    st.pydeck_chart(pdk.Deck(
        layers=[map_layer],
        initial_view_state=view_state,
        tooltip=tooltip_live,
        map_style=map_style_config
    ))

    st.divider()

    st.subheader('🤖 Analisis Zona Otomatis (AI Clustering)')
    st.caption(
        'Biarkan Machine Learning (K-Means) mengelompokan stasiun menjadi Zona Strategis')

    num_cluster = st.slider(
        'Jumlah Cluster', min_value=2, max_value=20, value=5)

    # get data for ml (just lat/lon)
    X = df_data_asli[['latitude', 'longitude']].dropna()

    # running k-means
    kmeans = KMeans(n_clusters=num_cluster, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # save 'label cluster' result to dataframe to show on map
    # we must make copy of df_data_asli to avoid warning
    df_clustered = df_data_asli.copy()
    df_clustered = df_clustered.loc[X.index]
    df_clustered['cluster_label'] = clusters

    # we change the colors by cluster id to makes the maps more vibrant
    # color dictionary (RGB) to 20 cluster first
    COLOR_PALETTE = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
        [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        [128, 0, 128], [0, 128, 128], [255, 128, 0], [
            255, 0, 128], [128, 255, 0],
        [0, 255, 128], [0, 128, 255], [128, 0, 255], [
            192, 192, 192], [64, 64, 64]
    ]

    def get_color(cluster_id):
        # get color from palette, if color is out of range return back to first color
        base_color = COLOR_PALETTE[cluster_id % len(COLOR_PALETTE)]
        return base_color + [160]  # alpha channel for transparency

    # apply colors to each rows
    df_clustered['color'] = df_clustered['cluster_label'].apply(get_color)

    # new maps layers (support dynamic colors)
    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_clustered,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius=5000,
        pickable=True,
        auto_highlight=True,
    )

    tooltip_cluster = {
        "html": "<b>{station_name}</b><br/>"
        "Zona (Cluster): <b>{cluster_label}</b></br>"
        "Kota: <b>{city}</b>"
    }
    st.pydeck_chart(pdk.Deck(
        layers=[cluster_layer],
        initial_view_state=view_state,
        map_style=map_style_config,
        tooltip=tooltip_cluster
    ))
    st.info(
        f'💡 Insight: Algoritma sudah membagi area menjadi : {num_cluster} zona. Coba geser slide untuk melihat bagaimana AI membagi area tersebut.')

    st.divider()

    st.subheader('📋 Profil Tiap Zona (Cluster Insight)')

    # keyword data science = groupBy Aggregation
    cluster_stats = df_clustered.groupby('cluster_label').agg({
        'station_name': 'count',
        'fuel_type': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
        'city': 'nunique',
        'state': 'nunique'
    }).reset_index()

    cluster_stats.columns = ['Zona ID', 'Total Stasiun',
                             'Dominan Fuel', 'Jml Kota', 'Jml N.Bagian']

    cluster_stats['Zona ID'] = cluster_stats['Zona ID'].apply(
        lambda x: f'Zona {x}')

    # show table
    st.dataframe(cluster_stats.style.background_gradient(
        subset=['Total Stasiun'], cmap='Reds'), use_container_width=True)

    # automatic story telling
    top_zone = cluster_stats.sort_values(
        by='Total Stasiun', ascending=False).iloc[0]
    st.success(f'**Zona Paling Padat :** {top_zone['Zona ID']} dengan **{top_zone['Total Stasiun']:,}** Stasiun. Di dominasi oleh bahan bakar **{top_zone['Dominan Fuel']}**. Zona ini mencangkup **{top_zone['Jml Kota']:,}** Kota dan **{top_zone['Jml N.Bagian']:,}** Negara Bagian')

    st.divider()

    # graph for visualizing fuel type distribution
    st.subheader('📊 Distribusi Jenis Bahan Bakar')

    # count of each fuel type
    fuel_counts = df_data_asli['fuel_type'].value_counts().reset_index()
    fuel_counts.columns = ['Jenis Bahan Bakar', 'Jumlah Stasiun']

    # make donut chart
    fig_fuel = px.pie(
        fuel_counts,
        values='Jumlah Stasiun',
        names='Jenis Bahan Bakar',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

    # table for show raw data
    with st.expander("🔍 Lihat Data Mentah BigQuery"):
        st.dataframe(df_data_asli, use_container_width=True)
