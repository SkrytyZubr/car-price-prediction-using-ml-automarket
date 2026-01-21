import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np 
import datetime

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="AutoMarket Pro | AI Valuation",
    page_icon="🚘",
    layout="wide"
)

# --- STAŁE (Statystyki Twojego modelu) ---
MAE_VALUE = 12987  # Wpisane na podstawie Twojego wyniku
R2_SCORE = 0.91    # Twój wynik

# --- ŁADOWANIE MODELU I DANYCH ---
@st.cache_resource
def load_data():
    files = ['model_ceny_aut.pkl', 'model_kolumny.pkl', 'mapa_marka_model.pkl']
    for f in files:
        if not os.path.exists(f):
            return None, None, None, None
    
    model = joblib.load('model_ceny_aut.pkl')
    cols = joblib.load('model_kolumny.pkl')
    brand_map = joblib.load('mapa_marka_model.pkl')
    
    if os.path.exists('baza_aut_clean.csv'):
        df = pd.read_csv('baza_aut_clean.csv')
    else:
        df = None
        
    return model, cols, df, brand_map

model, model_columns, df, brand_model_map = load_data()

# --- OBSŁUGA BŁĘDÓW ---
if model is None:
    st.error("⛔ BŁĄD KRYTYCZNY: Nie znaleziono plików modelu!")
    st.warning("Uruchom najpierw skrypt `analiza.py`, aby wygenerować model AI.")
    st.stop()

# --- SIDEBAR: KARTA MODELU ---
with st.sidebar:
    st.header("🧠 Karta Modelu AI")
    st.info("Parametry algorytmu uczącego")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R² Score", f"{R2_SCORE}", delta="Bardzo Wysoka")
    col_m2.metric("Błąd (MAE)", f"{MAE_VALUE/1000:.1f}k", delta_color="inverse")
    
    st.markdown("---")
    st.write("**Szczegóły techniczne:**")
    st.text(f"Silnik: XGBoost Regressor")
    st.text(f"Feature Engineering: TAK")
    st.text(f"(Wiek², Moc/Litr, Km/Rok)")
    st.text(f"Baza: {len(df) if df is not None else '30k+'} pojazdów")

# --- NAGŁÓWEK ---
st.title("🚘 System Wyceny Pojazdów (Feature Engineering + Tuning)")
st.markdown("Aplikacja wykorzystuje model XGBoost, który analizuje nieliniową utratę wartości oraz wysilenie silnika.")
st.markdown("---")

# --- ZAKŁADKI ---
tab1, tab2, tab3 = st.tabs(["💰 Kalkulator Wyceny", "📊 Analityka Rynku", "🏆 Rankingi i Detale"])

# ==========================================
# TAB 1: KALKULATOR
# ==========================================
with tab1:
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.subheader("⚙️ Konfiguracja Pojazdu")
        brands = sorted(list(brand_model_map.keys()))
        
        # Wybór Marki i Modelu
        s_brand = st.selectbox("Marka", brands)
        s_model = st.selectbox("Model", brand_model_map.get(s_brand, ["Inny"]))
        
        # Parametry
        current_year = datetime.datetime.now().year
        s_year = st.slider("Rok produkcji", 2000, current_year, 2020)
        s_mileage = st.number_input("Przebieg (km)", 0, 500000, 150000, step=5000)
        s_hp = st.number_input("Moc (KM)", 50, 800, 150)
        
        # KLUCZOWE: Pojemność jest potrzebna do obliczenia hp_per_liter
        s_capacity = st.number_input("Pojemność (cm3)", 500, 8000, 1995, step=100)
        
        s_fuel = st.selectbox("Paliwo", ["Diesel", "Benzyna", "Hybryda", "Elektryczny"])
        s_trans = st.selectbox("Skrzynia", ["Manualna", "Automatyczna"])
        
        # Ukryte (domyślne)
        s_type = "Osoba prywatna"
        s_region = "Mazowieckie"
    
    with col_r:
        st.subheader("💸 Wynik Wyceny")
        st.info("Kliknij przycisk, aby przetworzyć dane i uruchomić model.")
        
        if st.button("OSZACUJ WARTOŚĆ", type="primary", use_container_width=True):
            
            # 1. Przygotowanie pustego DataFrame
            input_data = pd.DataFrame(columns=model_columns)
            input_data.loc[0] = 0 
            
            # 2. Wypełnienie danych podstawowych
            input_data['year'] = s_year
            input_data['mileage_km'] = s_mileage
            input_data['horsepower_hp'] = s_hp
            input_data['engine_capacity_cm3'] = s_capacity
            
            # --- 3. FEATURE ENGINEERING (Matematyka zgodna z analiza.py) ---
            # To musi być identyczne jak w procesie treningu!
            car_age = current_year - s_year
            input_data['car_age'] = car_age
            input_data['car_age_squared'] = car_age ** 2
            
            # Zabezpieczenie przed dzieleniem przez zero (+1)
            input_data['km_per_year'] = s_mileage / (car_age + 1)
            
            # Zabezpieczenie pojemności
            cap_liter = s_capacity / 1000 if s_capacity > 0 else 2.0
            input_data['hp_per_liter'] = s_hp / cap_liter
            # ---------------------------------------------------------------
            
            # 4. One-Hot Encoding
            for feat in [f'brand_{s_brand}', f'model_{s_model}', f'fuel_type_{s_fuel}', 
                         f'transmission_{s_trans}', f'seller_type_{s_type}', f'location_region_{s_region}']:
                if feat in input_data.columns: input_data[feat] = 1
            
            try:
                # Model zwraca logarytm -> odwracamy
                pred_log = model.predict(input_data)[0]
                pred_pln = np.expm1(pred_log)
                
                # --- WYNIK I MARGINES BŁĘDU ---
                st.success(f"### Szacowana cena: {pred_pln:,.0f} PLN")
                
                lower_bound = max(0, pred_pln - MAE_VALUE)
                upper_bound = pred_pln + MAE_VALUE
                error_pct = (MAE_VALUE / pred_pln) * 100
                
                c_err1, c_err2 = st.columns(2)
                with c_err1:
                    st.warning(f"📉 **Przedział rynkowy:**\n{lower_bound:,.0f} PLN — {upper_bound:,.0f} PLN")
                with c_err2:
                    st.info(f"📊 **Margines błędu:**\n+/- {MAE_VALUE:,.0f} PLN (ok. {error_pct:.1f}%)")

                st.caption(f"Specyfikacja: {s_brand} {s_model}, {s_year}, {s_hp} KM, {s_capacity} cm3")
                # st.progress(min(pred_pln / 400000, 1.0))
                
            except Exception as e:
                st.error(f"Błąd obliczeń: {e}")
                st.write("Wskazówka: Sprawdź czy pliki .pkl są aktualne.")

# ==========================================
# TAB 2: ANALITYKA (OGÓLNA)
# ==========================================
with tab2:
    if df is not None:
        st.header("Analiza czynników cenowych")
        
        # 1. Feature Importance (Agregowane)
        st.subheader("1. Co najbardziej wpływa na cenę?")
        importances = model.feature_importances_
        feature_names = model_columns
        
        feature_groups = {}
        for name, imp in zip(feature_names, importances):
            if "brand_" in name: key = "Marka"
            elif "model_" in name: key = "Model Auta"
            elif "fuel_" in name: key = "Paliwo"
            elif "transmission_" in name: key = "Skrzynia Biegów"
            elif "year" in name or "age" in name: key = "Wiek / Rocznik"
            elif "power" in name or "hp" in name: key = "Moc Silnika"
            elif "mileage" in name or "km_" in name: key = "Przebieg"
            elif "capacity" in name or "liter" in name: key = "Pojemność Silnika"
            else: key = "Inne"
            feature_groups[key] = feature_groups.get(key, 0) + imp

        df_imp = pd.DataFrame(list(feature_groups.items()), columns=["Cecha", "Wpływ"])
        df_imp = df_imp.sort_values("Wpływ", ascending=True)
        
        fig_imp = px.bar(df_imp, x="Wpływ", y="Cecha", orientation='h', 
                         title="Ranking ważności cech (wg modelu XGBoost)",
                         color="Wpływ", color_continuous_scale="Blues")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.divider()

        # 2. Przebieg vs Cena
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2. Wpływ przebiegu")
            fig_miles = px.scatter(df, x="mileage_km", y="price", color="fuel_type",
                                   title="Przebieg vs Cena (dla różnych paliw)",
                                   labels={"mileage_km": "Przebieg (km)", "price": "Cena (PLN)"},
                                   range_x=[0, 400000], range_y=[0, 300000], opacity=0.3)
            st.plotly_chart(fig_miles, use_container_width=True)
            
        with c2:
            st.subheader("3. Krzywa amortyzacji")
            
            # FILTROWANIE: Bierzemy tylko lata 2000 - 2026
            mask = (df['year'] >= 2000) & (df['year'] <= 2026)
            df_line = df[mask].groupby('year')['price'].median().reset_index()
            
            fig_line = px.line(df_line, x='year', y='price', markers=True,
                            title="Mediana ceny wg rocznika (2000-2026)")
            st.plotly_chart(fig_line, use_container_width=True)

    else:
        st.warning("Brak danych do wykresów.")

# ==========================================
# TAB 3: RANKINGI I DETALE
# ==========================================
with tab3:
    if df is not None:
        st.header("🏆 Benchmarking Marek")
        st.write("Porównanie producentów pod kątem ekonomii i pozycjonowania.")
        
        # 1. Ekonomia Mocy (Złoty za KM)
        st.subheader("1. Koszt Mocy (Cena za 1 KM)")
        df_power = df.groupby('brand').agg({'price': 'mean', 'horsepower_hp': 'mean'}).reset_index()
        df_power['pln_per_hp'] = df_power['price'] / df_power['horsepower_hp']
        df_power = df_power.sort_values('pln_per_hp', ascending=True)
        
        fig_power = px.bar(df_power, x='brand', y='pln_per_hp', color='pln_per_hp',
                           title="Ranking: Która marka oferuje najtańsze konie mechaniczne?",
                           labels={'pln_per_hp': 'Cena za 1 KM (PLN)', 'brand': ''},
                           color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_power, use_container_width=True)
        
        st.divider()
        
        # 2. Violin Plot i Struktura Paliw
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("2. Struktura Napędów")
            df_fuel = df.groupby(['brand', 'fuel_type']).size().reset_index(name='count')
            fig_fuel = px.bar(df_fuel, x="brand", y="count", color="fuel_type",
                              title="Jakie paliwo dominuje u danej marki?",
                              labels={'count': 'Liczba ofert', 'brand': ''})
            st.plotly_chart(fig_fuel, use_container_width=True)
            
        with c4:
            st.subheader("3. Rozkład Cen (Violin Plot)")
            sorted_brands = df.groupby('brand')['price'].median().sort_values(ascending=False).index
            fig_violin = px.violin(df, y="brand", x="price", box=True, points=False,
                                   category_orders={"brand": sorted_brands},
                                   range_x=[0, 400000], color="brand",
                                   title="Gęstość ofert cenowych")
            fig_violin.update_layout(showlegend=False, yaxis_title="")
            st.plotly_chart(fig_violin, use_container_width=True)
        
        st.divider()

        # 3. Tabela Top Modeli
        st.subheader("4. Modele Skrajne (Średnia cena)")
        df_models = df.groupby(['brand', 'model'])['price'].mean().reset_index()
        df_models = df_models[df_models['model'] != "Inny"]
        
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("💎 **Top 10 Najdroższych**")
            top = df_models.sort_values('price', ascending=False).head(10)
            st.dataframe(top.style.format({"price": "{:,.0f} PLN"}), use_container_width=True)
            
        with c6:
            st.markdown("💸 **Top 10 Najtańszych**")
            bot = df_models.sort_values('price', ascending=True).head(10)
            st.dataframe(bot.style.format({"price": "{:,.0f} PLN"}), use_container_width=True)