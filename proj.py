import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ustawienia regionalne, PL (nazwy dni tygodnia i miesięcy)
import locale
try:
    locale.setlocale(locale.LC_TIME, 'pl_PL.UTF-8')
except:
    locale.setlocale(locale.LC_TIME, 'C')

# link do tła
image_url = "https://www.dropbox.com/scl/fi/exssbqu7d76al00osw1d0/bustlo.png?rlkey=j9r8yi01w442acm0bj4efp5sk&st=g88qbz90&raw=1"

# konfiguracja strony
#layout wide = szeroki układ, bo domyślnie brzydko centrował
st.set_page_config(page_title="projekt WSB AP", layout="wide")

# Styl tła i treści + styl przycisków w panelu bocznym
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.85);
        padding: 3rem;
        border-radius: 15px;
    }}
    .stMarkdown, .stDataFrame, .stPlotlyChart {{
        color: white !important;
    }}

    /* Styl kafelków przycisków*/
    div.stButton > button {{
        background-color: #111111;
        color: white;
        width: 100%;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 1.5rem;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        text-align: center;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    div.stButton > button:hover {{
        background-color: #222222;
        color: #7B68EE;
    }}
    div.stButton > button:focus,
    div.stButton > button:active {{
        background-color: #222222 !important;
        color: #00bfff !important;
        border: 1px solid #444 !important;
        outline: none !important;
        box-shadow: none !important;
    }}
        .stTable, .stTable th, .stTable td {{
        background-color: rgba(0, 0, 0, 0.85) !important;
        color: white !important;
    }}
    .stTable {{
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    </style>
""", unsafe_allow_html=True)

torun = pd.read_csv('data/raport4_hzpkm_t.csv')
otwock = pd.read_csv('data/raport4_hzpkm_o.csv')
koszalin = pd.read_csv('data/raport4_hzpkm_k.csv')

# Otwock – format: '30-03-2025 23:30:00'
try:
    otwock["data"] = pd.to_datetime(otwock["data"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
except Exception as e:
    print("Błąd konwersji daty w Otwocku:", e)

# Koszalin i Toruń – zwykły format ISO
try:
    koszalin["data"] = pd.to_datetime(koszalin["data"], errors="coerce")
except Exception as e:
    print("Błąd konwersji daty w Koszalinie:", e)

try:
    torun["data"] = pd.to_datetime(torun["data"], errors="coerce")
except Exception as e:
    print("Błąd konwersji daty w Toruniu:", e)

# sidebar z logo
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 5px;'>
        <img src="https://www.dropbox.com/scl/fi/gxviulsghhy3pa486v3ss/logo-projekt-male.png?rlkey=ljfwd90cfqbfrfdesd810n250&st=488e2nzp&raw=1" width="220">
    </div>
""", unsafe_allow_html=True)

# lista stron
pages = ["O projekcie", "Toruń", "Otwock", "Koszalin", "Top 10 spóźnialskich", "Predykcja opóźnień"]

# start 
if "page" not in st.session_state:
    st.session_state.page = pages[0]

# przyciski - podświetlenie, klasa
for p in pages:
    if st.session_state.page == p:
        with st.sidebar:
            st.markdown('<div class="sidebar-active">', unsafe_allow_html=True)
            if st.button(p, key=p):
                st.session_state.page = p
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.sidebar.button(p, key=p):
            st.session_state.page = p

# aktywna strona, tytuł
page = st.session_state.page
st.title(page)

# przerzucenie modelu do cache'a żeby szybciej ładował stronę
#łączenie, dopisywanie kolumny, model lasów losowych 
@st.cache_data
def przygotuj_dane(torun, otwock, koszalin):
    torun["miasto"] = "Toruń"
    otwock["miasto"] = "Otwock"
    koszalin["miasto"] = "Koszalin"
    df = pd.concat([torun, otwock, koszalin], ignore_index=True)

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    dni_polskie = {
        0: "Poniedziałek", 1: "Wtorek", 2: "Środa",
        3: "Czwartek", 4: "Piątek", 5: "Sobota", 6: "Niedziela"
    }
    df["dzień_tygodnia"] = df["data"].dt.dayofweek.map(dni_polskie)
    df["spóźniony_5min"] = (df["opóźnienie[s]"] > 300).astype(int)
    df = df.dropna(subset=["linia", "dzień_tygodnia", "spóźniony_5min"])
    df["linia"] = df["linia"].astype(str).str.strip()
    return df

@st.cache_data
def zakoduj_dane(df):
    le_linia = LabelEncoder()
    le_dzien = LabelEncoder()
    df["linia_encoded"] = le_linia.fit_transform(df["linia"])
    df["dzien_encoded"] = le_dzien.fit_transform(df["dzień_tygodnia"])
    return df, le_linia, le_dzien

@st.cache_resource
def trenuj_model(df):
    A = df[["linia_encoded", "dzien_encoded"]]
    B = df["spóźniony_5min"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(A, B)
    return model

# Sekcja: O projekcie
if page == "O projekcie":
    st.markdown("""
    # Analiza i predykcja opóźnień komunikacji miejskiej

    W poniższym projekcie przeprowadzona została analiza opóźnień pojazdów względem rozkładu jazdy. Dane pochodzą z meldunków, które pojazdy przesyłają, znajdując się w obrębie wybranych, osygnalizowanych skrzyżowań zlokalizowanych na terenie trzech przykładowych miast: **Toruń**, **Otwock** i **Koszalin**.  

    Projekt powstał na potrzeby zaliczenia studiów podyplomowych na kierunku Data Scientist — Analityk Danych i ma charakter edukacyjny, nie jest produktem komercyjnym.  

    **Źródła danych:**

    - dane źródłowe w postaci plików csv pobrano z aplikacji należących do miejskich zarządów dróg. Nazwy miast i ulic zostały zmienione, a dane poddano wstępnej selekcji polegającej na wybraniu jedynie części funkcjonujących linii oraz krótkiego przedziału czasu (marzec 2025),
    - grafika stworzona za pomocą narzędzia AI ([Microsoft Copilot](https://copilot.microsoft.com)),
    - zdjęcie w tle aplikacji — fot. Paweł Przymanowski, wykorzystane za zgodą autora. Zdjęcie przedstawia pojazd floty GAiT, edytowane za pomocą narzędzi AI ([ChatGPT](https://chatgpt.com)).

    **Zastosowane technologie:**

    Projekt został wykonany w języku Python 3 z wykorzystaniem bibliotek takich jak: streamlit (aplikacja webowa), pandas, numpy (przetwarzanie danych i obliczenia), plotly (wizualizacje) oraz scikit-learn (model uczenia maszynowego).  

    **Autor:** Aleksandra Pastwa, nr albumu: 27750  
    """)
# Sekcje miast
elif page in ["Toruń", "Otwock", "Koszalin"]:
    df = torun if page == "Toruń" else otwock if page == "Otwock" else koszalin
    st.subheader("Wybierz parametry filtracji")
    col1, col2 = st.columns(2)
    with col1:
        linie = st.multiselect("Wybierz linie:", sorted(df['linia'].unique()))
    with col2:
        priorytet = st.selectbox("Czy nadano priorytet?", ["Wszystkie", "TAK", "NIE"])
    if linie:
        df = df[df['linia'].isin(linie)]
    if priorytet != "Wszystkie":
        df = df[df['priorytet'].str.upper() == priorytet]

    st.markdown("### Dane filtrowane:")
    st.dataframe(df, use_container_width=True)

	# średnie opóźnienie na liniach
    st.markdown("### Średnie opóźnienie dla linii")
    df_linie = df.groupby("linia")["opóźnienie[s]"].mean().reset_index()
    df_linie = df_linie.sort_values(by="opóźnienie[s]", ascending=False)
    colors = px.colors.sequential.Magma
    num_colors = len(df_linie)

	# ydłużenie palety jeśli trzeba
    color_cycle = (colors * ((num_colors // len(colors)) + 1))[:num_colors]

    fig1 = px.bar(
		df_linie,
		x="linia",
		y="opóźnienie[s]",
		labels={"opóźnienie[s]": "Średnie opóźnienie (s)", "linia": "Linia"},
		title="Średnie opóźnienie dla linii"
	)
    fig1.update_traces(marker_color=color_cycle)
    fig1.update_layout(showlegend=False)
    fig1.update_xaxes(type='category')
    st.plotly_chart(fig1, use_container_width=True)

	# średnie opóźnienie na skrzyżowaniach
    st.markdown("### Średnie opóźnienie dla skrzyżowania")
    df_skrzyz = df.groupby("skrzyżowanie")["opóźnienie[s]"].mean().reset_index()
    df_skrzyz = df_skrzyz.sort_values(by="opóźnienie[s]", ascending=False).head(20)

    colors2 = px.colors.sequential.Magma_r
    num_colors2 = len(df_skrzyz)
    color_cycle2 = (colors2 * ((num_colors2 // len(colors2)) + 1))[:num_colors2]

    fig2 = px.bar(
		df_skrzyz,
		x="skrzyżowanie",
		y="opóźnienie[s]",
		labels={"opóźnienie[s]": "Średnie opóźnienie (s)", "skrzyżowanie": "Skrzyżowanie"},
		title="Średnie opóźnienie dla skrzyżowania"
	)
    fig2.update_traces(marker_color=color_cycle2)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# reszta
elif page == "Top 10 spóźnialskich":
    # kolumna 'miasto', żeby wiedzieć skąd dany wiersz pochodzi
    torun["miasto"] = "Toruń"
    otwock["miasto"] = "Otwock"
    koszalin["miasto"] = "Koszalin"

    df_all = pd.concat([torun, otwock, koszalin], ignore_index=True)

    # Sortuj od max opóźnienia + koronki i 7 niców
    top10 = df_all.sort_values(by="opóźnienie[s]", ascending=False).head(10).copy()
    korony = ["👑 Złota", "🥈 Srebrna", "🥉 Brązowa"] + [" "] * 7
    top10["nagroda"] = korony

    # kolumny do pokazania, zmiana nazwy
    top10_display = top10[["nagroda", "opóźnienie[s]", "linia", "pojazd", "data", "miasto"]]
    top10_display.columns = ["🏆 Nagroda", "Opóźnienie [s]", "Linia", "Numer pojazdu", "Data przejazdu", "Miasto"]
    top10_display = top10_display.reset_index(drop=True)
    st.markdown("### Top 10 spóźnionych przejazdów!")
    st.table(top10_display)


elif page == "Predykcja opóźnień":
    st.markdown("""
    ###  Czy chcesz wiedzieć, jaka jest szansa, że twój autobus przyjedzie spóźniony powyżej 5 min? 
    Wybierz linię oraz dzień tygodnia i przekonaj się sam! 🚌🐢
    """)

    # przygotowanie danych
    df_all1 = przygotuj_dane(torun, otwock, koszalin)
    df_all1, le_linia, le_dzien = zakoduj_dane(df_all1)
    model = trenuj_model(df_all1)

    # UI – wybory
    col1, col2, col3 = st.columns(3)
    with col1:
        miasto_input = st.selectbox("Wybierz miasto:", sorted(df_all1["miasto"].unique()))
    with col2:
        linie_dla_miasta = sorted(df_all1[df_all1["miasto"] == miasto_input]["linia"].unique())
        linia_input = st.selectbox("Wybierz linię:", linie_dla_miasta)
    with col3:
        dzien_input = st.selectbox("Wybierz dzień tygodnia:", sorted(df_all1["dzień_tygodnia"].unique()))

    # Predykcja
    linia_enc = le_linia.transform([linia_input])[0]
    dzien_enc = le_dzien.transform([dzien_input])[0]
    pred_prob = model.predict_proba([[linia_enc, dzien_enc]])[0][1]
    procent = round(pred_prob * 100, 2)

    st.markdown(f"## Szansa na spóźnienie powyżej 5 minut dla linii **{linia_input}** w dzień **{dzien_input}** wynosi: **{procent}%**")

    # rozklad opóźnień wykres kołowy
    df_filtered = df_all1[
        (df_all1["linia"] == linia_input) &
        (df_all1["dzień_tygodnia"] == dzien_input) &
        (df_all1["miasto"] == miasto_input)
    ]

    if not df_filtered.empty:
        def kategoria_opoznienia(sek):
            if sek < 0:
                return "Przed czasem"
            elif 0 <= sek < 120:
                return "Spóźniony <2 min"
            elif 120 <= sek < 300:
                return "Spóźniony 2–5 min"
            elif 300 <= sek < 600:
                return "Spóźniony 5–10 min"
            else:
                return "Spóźniony >10 min"

        df_filtered["kategoria"] = df_filtered["opóźnienie[s]"].apply(kategoria_opoznienia)

        rozkład = df_filtered["kategoria"].value_counts(normalize=True).sort_index()
        rozkład_procent = rozkład * 100

        # statystyki tabelka
        opoznienia = df_filtered["opóźnienie[s]"].to_numpy()
        srednia = np.mean(opoznienia)
        mediana = np.median(opoznienia)
        std_dev = np.std(opoznienia)

        stats_df = pd.DataFrame({
            "Statystyka": ["Średnia opóźnienia [s]", "Mediana opóźnienia [s]", "Odchylenie standardowe [s]"],
            "Wartość": [round(srednia, 2), round(mediana, 2), round(std_dev, 2)]
        })

        col1, col2 = st.columns([2, 2])

        with col1:
            fig_pie = px.pie(
                names=rozkład_procent.index,
                values=rozkład_procent.values,
                title=f"Rozkład opóźnień – linia {linia_input}, {dzien_input}, {miasto_input}",
                color_discrete_sequence=["#6a0dad", "#d81b60", "#ff7043", "#d32f2f"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("### Statystyki opóźnień")
            st.table(stats_df)
    else:
        st.warning("Brak danych dla tej kombinacji (linia, dzień, miasto) – nie można wygenerować wykresu.")

