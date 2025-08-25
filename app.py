import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(page_title="Gas Dashboard", layout="wide")
st.title("\U0001F4CA EUA Analytics Dashboard")

# rendre la barre d'onglets scrollable si elle dépasse en largeur
st.markdown("""
<style>
.stTabs [role="tablist"]{
  overflow-x:auto !important;
  white-space:nowrap !important;
}
.stTabs [role="tab"]{
  flex:0 0 auto !important;
}
</style>
""", unsafe_allow_html=True)

# --- chemins robustes ---
APP_DIR = Path(__file__).resolve().parent
file_path = APP_DIR / "Gas storages.xlsx"   # évite les surprises de CWD

# 1) ➕ ajoute l’onglet
tabs = st.tabs([
    "📦 Stocks",
    "💰 Prix (EUA/TTF)",
    "📈 Stratégies RSI / StochRSI",
    "📊 Open Interest",
    "⏩ Forward Curve",
    "🌡️ Temp & HDD"   # <— nouvel onglet
])

st.caption(f"Debug: {len(tabs)} onglets créés")

# === 1. Onglet STOCKS ===
with tabs[0]:
    st.header("Stockages de gaz - par pays")

    start_year = 2020
    end_year = 2025

    columns_mapping = [
        'Date', 'Europe Gas Storage (TWh)', 'US DOE estimated storage',
        'UK Gas Storage (TWh)', 'Germany Gas Storage (TWh)', 'Netherlands Gas Storage (TWh)'
    ]

    colors = {2020:'blue', 2021:'orange', 2022:'purple', 2023:'yellow', 2024:'green', 2025:'red'}

    # bouton manuel si besoin
    if st.button("🔄 Forcer la mise à jour des données"):
        st.cache_data.clear()
        st.rerun()

    # === clé de cache liée au fichier ===
    @st.cache_data(show_spinner=False)
    def load_stock_data(xlsx_path: Path, file_version: float):
        # file_version = os.path.getmtime(xlsx_path) -> utilisé pour invalider le cache
        df = pd.read_excel(xlsx_path, sheet_name="Stocks", header=None, skiprows=6)
        df = df.iloc[:, :6]
        df.columns = columns_mapping
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        # normalisation numérique
        for c in columns_mapping[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna()

    file_mtime = os.path.getmtime(file_path)
    df_stock = load_stock_data(file_path, file_mtime)

    # petite info de contrôle
    st.caption(f"Dernière date lue : **{df_stock['Date'].max().date()}**  (mtime: {int(file_mtime)})")

    country_map = {
        'Europe Gas Storage (TWh)': 'Europe',
        'US DOE estimated storage': 'US',
        'UK Gas Storage (TWh)': 'UK',
        'Germany Gas Storage (TWh)': 'Germany',
        'Netherlands Gas Storage (TWh)': 'Netherlands'
    }
    selected_country = st.selectbox("Choisir un pays :", list(country_map.keys()))

    series = df_stock[['Date', selected_country]].dropna()
    series['Value'] = pd.to_numeric(series[selected_country], errors='coerce')
    series = series[series['Date'].dt.year >= start_year].dropna()

    range_data = series[series['Date'].dt.year <= 2024].copy()
    range_data['DOY'] = range_data['Date'].dt.dayofyear

    all_years = []
    for year in range(2020, 2025):
        yearly = range_data[range_data['Date'].dt.year == year].copy()
        yearly = yearly.groupby('DOY')['Value'].mean().reindex(np.arange(1, 367)).interpolate()
        all_years.append(yearly.values)

    all_years_array = np.vstack(all_years)
    min_vals = np.nanmin(all_years_array, axis=0)
    max_vals = np.nanmax(all_years_array, axis=0)
    mean_vals = np.nanmean(all_years_array, axis=0)

    full_doy = np.arange(1, 367)
    mois = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    mois_jours = [15,45,75,105,135,165,195,225,255,285,315,345]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_doy, y=min_vals, mode='lines', line=dict(color='lightgray'), showlegend=False))
    fig.add_trace(go.Scatter(x=full_doy, y=max_vals, mode='lines', fill='tonexty',
                             line=dict(color='lightgray'), name='Min-Max 2020–2024',
                             fillcolor='rgba(128,128,128,0.3)'))
    fig.add_trace(go.Scatter(x=full_doy, y=mean_vals, mode='lines', name='Moyenne 2020–2024',
                             line=dict(color='black', dash='dash')))

    for year in range(start_year, end_year + 1):
        yearly = series[series['Date'].dt.year == year].copy()
        if not yearly.empty:
            yearly['DOY'] = yearly['Date'].dt.dayofyear
            fig.add_trace(go.Scatter(
                x=yearly['DOY'], y=yearly['Value'], mode='lines', name=str(year),
                line=dict(width=2 if year >= 2023 else 1), opacity=1.0 if year >= 2023 else 0.4
            ))

    fig.update_layout(
        title=f"{country_map[selected_country]} - Stockage de gaz (TWh)",
        xaxis=dict(title="Mois", tickmode='array', tickvals=mois_jours, ticktext=mois),
        yaxis_title="TWh", legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=50, b=40), height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# === 2. Onglet PRIX ===
with tabs[1]:
    st.header("Prix du marché - EUA & TTF")
    df_prices = pd.read_excel(file_path, sheet_name="Prices", skiprows=6)
    df_prices.columns = ['Date', 'EUA', 'TTF']
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], errors='coerce')
    df_prices = df_prices.dropna(subset=['Date'])
    df_prices['Year'] = df_prices['Date'].dt.year
    df_prices = df_prices[df_prices['Year'].between(2021, 2025)]
    df_prices['DayOfYear'] = df_prices['Date'].dt.dayofyear

    def seasonal_price_plotly(df, col, ylabel, exclude=None):
        fig = go.Figure()
        for year in sorted(df['Year'].unique()):
            if exclude and year in exclude:
                continue
            data = df[df['Year'] == year]
            fig.add_trace(go.Scatter(
                x=data['DayOfYear'],
                y=data[col],
                mode='lines',
                name=str(year),
                opacity=1.0 if year >= 2023 else 0.3
            ))
        ticks = [pd.Timestamp(2022, m, 1).dayofyear for m in range(1, 13)]
        labels = [calendar.month_abbr[m] for m in range(1, 13)]
        fig.update_layout(
            title=f"{col} - Seasonal Daily Pattern",
            xaxis=dict(title="Month", tickmode='array', tickvals=ticks, ticktext=labels),
            yaxis_title=ylabel,
            legend_title="Année",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    seasonal_price_plotly(df_prices, 'EUA', "Price (€/tCO2)")
    seasonal_price_plotly(df_prices, 'TTF', "Price (€/MWh)", exclude=[2021, 2022])

# === 3. STRATÉGIES RSI ===
with tabs[2]:
    st.header("Stratégies techniques sur le marché EUA")

    df = pd.read_excel(file_path, sheet_name="Prices", skiprows=6, usecols="A,B")
    df.columns = ['Date', 'EUA']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['EUA'] = pd.to_numeric(df['EUA'], errors='coerce')
    df = df.dropna().set_index('Date')

    delta = df['EUA'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi = df['RSI']
    stochrsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    df['StochRSI'] = stochrsi

    def run_strategy(df, long_cond, short_cond):
        trades = []
        position = None
        for date, row in df.iterrows():
            if not (2021 <= date.year <= 2025 and 4 <= date.month <= 9):
                continue
            price = row['EUA']
            if position is None:
                if long_cond(row):
                    position = {'type': 'long', 'entry_price': price}
                elif short_cond(row):
                    position = {'type': 'short', 'entry_price': price}
            elif position:
                entry = position['entry_price']
                if position['type'] == 'long':
                    if price >= entry + 2:
                        trades.append({'date': date, 'pnl': 2, 'type': 'long'})
                        position = None
                    elif price <= entry - 1:
                        trades.append({'date': date, 'pnl': -1, 'type': 'long'})
                        position = None
                elif position['type'] == 'short':
                    if price <= entry - 2:
                        trades.append({'date': date, 'pnl': 2, 'type': 'short'})
                        position = None
                    elif price >= entry + 1:
                        trades.append({'date': date, 'pnl': -1, 'type': 'short'})
                        position = None
        tdf = pd.DataFrame(trades).set_index('date').sort_index()
        tdf['PnL €'] = tdf['pnl'] * 100000
        tdf['Cumulative PnL'] = tdf['PnL €'].cumsum()
        tdf['Year'] = tdf.index.year
        return tdf

    trades_rsi = run_strategy(df, lambda r: r['RSI'] < 30, lambda r: r['RSI'] > 70)
    trades_stoch = run_strategy(df, lambda r: r['StochRSI'] < 0.2, lambda r: r['StochRSI'] > 0.8)

    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("Stochastic RSI (14)")
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df['StochRSI'], mode='lines', name='StochRSI', line_color='orange'))
    fig_stoch.add_hline(y=0.8, line_dash="dash", line_color="red")
    fig_stoch.add_hline(y=0.2, line_dash="dash", line_color="green")
    st.plotly_chart(fig_stoch, use_container_width=True)

    st.subheader("Cumulative PnL des stratégies")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=trades_rsi.index, y=trades_rsi['Cumulative PnL'], name='RSI Strategy'))
    fig_pnl.add_trace(go.Scatter(x=trades_stoch.index, y=trades_stoch['Cumulative PnL'], name='StochRSI Strategy'))
    fig_pnl.update_layout(yaxis_title="Cumulative PnL (€)")
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.subheader("PnL Annuel par stratégie")
    annual_rsi = trades_rsi.groupby('Year')['PnL €'].sum()
    annual_stoch = trades_stoch.groupby('Year')['PnL €'].sum()

    x = np.arange(len(annual_rsi.index))
    bar_width = 0.35
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=annual_rsi.index - 0.2, y=annual_rsi.values, name='RSI Strategy'))
    fig_bar.add_trace(go.Bar(x=annual_stoch.index + 0.2, y=annual_stoch.values, name='StochRSI Strategy'))
    fig_bar.update_layout(barmode='group', xaxis_title='Année', yaxis_title='PnL (€)')
    st.plotly_chart(fig_bar, use_container_width=True)

# === 4. Onglet OPEN INTEREST ===
with tabs[3]:
    st.header("Analyse des contrats EUA (Open Interest)")

    file_path_oi = APP_DIR / "EUA OI & forward.xlsx"
    sheet_name = "Sheet2"

    if not file_path_oi.exists():
        st.error(f"Fichier introuvable : {file_path_oi}")
    else:
        # titres en ligne 1
        titles_row = pd.read_excel(file_path_oi, sheet_name=sheet_name, header=None, nrows=1)
        titles = titles_row.iloc[0, 1:].astype(str).tolist()

        # données à partir de la ligne 5
        df_oi = pd.read_excel(file_path_oi, sheet_name=sheet_name, skiprows=4, header=None)
        df_oi.columns = ["Date"] + titles

        # ⚠️ ignorer la toute première ligne de données (ligne 5 Excel)
        df_oi = df_oi.iloc[1:, :]

        # conversions + tri
        df_oi["Date"] = pd.to_datetime(df_oi["Date"], errors="coerce")
        for col in df_oi.columns[1:]:
            df_oi[col] = pd.to_numeric(df_oi[col], errors="coerce")
        df_oi = df_oi.dropna(subset=["Date"]).sort_values("Date")

        # superposé
        fig_oi = go.Figure()
        for col in df_oi.columns[1:]:
            fig_oi.add_trace(go.Scatter(x=df_oi["Date"], y=df_oi[col], mode="lines", name=col))
        fig_oi.update_layout(
            title="Historique des contrats (Open Interest) - Superposé",
            xaxis_title="Date", yaxis_title="Open Interest",
            legend=dict(orientation="h"),
            margin=dict(l=40, r=40, t=50, b=40), height=600
        )
        st.plotly_chart(fig_oi, use_container_width=True)

        # individuels
        st.subheader("Graphiques par contrat")
        for col in df_oi.columns[1:]:
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=df_oi["Date"], y=df_oi[col], mode="lines", name=col))
            fig_ind.update_layout(
                title=f"Historique Open Interest - {col}",
                xaxis_title="Date", yaxis_title="Open Interest",
                margin=dict(l=40, r=40, t=50, b=40), height=400
            )
            st.plotly_chart(fig_ind, use_container_width=True)


# === 5. Onglet FORWARD CURVE ===
with tabs[4]:
    st.header("EUA Forward Curve")

    file_path_fwd = APP_DIR / "EUA OI & forward.xlsx"
    # tu peux cibler par nom...
    sheet_name_fwd = "Sheet3"
    # ...ou par index au cas où (3e feuille = index 2)
    # sheet_name_fwd = 2

    if not file_path_fwd.exists():
        st.error(f"Fichier introuvable : {file_path_fwd}")
    else:
        # Col A = contrats, Col B = valeurs ; données dès la ligne 2
        df_fwd = pd.read_excel(file_path_fwd, sheet_name=sheet_name_fwd, header=None, skiprows=1, usecols=[0,1])
        df_fwd.columns = ["Contract", "Value"]
        df_fwd["Value"] = pd.to_numeric(df_fwd["Value"], errors="coerce")
        df_fwd = df_fwd.dropna(subset=["Value"]).reset_index(drop=True)

        # Renomme en M1, M2, M3, ...
        df_fwd["Contract"] = [f"M{i+1}" for i in range(len(df_fwd))]

        fig_fwd = go.Figure()
        fig_fwd.add_trace(go.Scatter(
            x=df_fwd["Contract"], y=df_fwd["Value"],
            mode="lines+markers", name="Forward Curve"
        ))
        fig_fwd.update_layout(
            title="EUA Forward Curve",
            xaxis_title="Maturité (M1, M2, …)",
            yaxis_title="Prix (€/tCO2)",
            margin=dict(l=40, r=40, t=50, b=40), height=500
        )
        st.plotly_chart(fig_fwd, use_container_width=True)

# 2) 🌡️ Onglet TEMP & HDD — organisé par pays
with tabs[5]:
    st.header("Températures saisonnières & HDD mensuels")
    st.caption("🧪 Debug: l’onglet est bien monté")

    try:
        # ---------- Chargement des données ----------
        hdd_file = APP_DIR / "HDD EUA.xlsx"
        if not hdd_file.exists():
            st.error(f"Fichier introuvable : {hdd_file}")
            st.stop()

        # mapping des colonnes (feuille "Historical temp & HDD")
        temp_cols = {
            "France": "Last Price",
            "UK": "Last Price.1",
            "Belgium": "Last Price.2",
            "Netherlands": "Last Price.3",
            "Germany": "Last Price.4",
            "Poland": "Last Price.5",
        }
        hdd_cols = {
            "France": "Unnamed: 7",
            "UK": "Unnamed: 8",
            "Belgium": "Unnamed: 9",
            "Netherlands": "Unnamed: 10",
            "Germany": "Unnamed: 11",
            "Poland": "Unnamed: 12",
        }

        @st.cache_data(show_spinner=False)
        def load_temp_hdd(xlsx_path: Path, file_version: float):
            df = pd.read_excel(xlsx_path, sheet_name="Historical temp & HDD", skiprows=5)
            # drop de la 1re ligne ("PX_LAST")
            df = df.drop(index=0).reset_index(drop=True)
            df = df.rename(columns={"Unnamed: 0": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df

        file_mtime = os.path.getmtime(hdd_file)
        df_th = load_temp_hdd(hdd_file, file_mtime)

        # Vérif colonnes pour éviter KeyError silencieux
        missing_t = [c for c in temp_cols.values() if c not in df_th.columns]
        missing_h = [c for c in hdd_cols.values() if c not in df_th.columns]
        if missing_t or missing_h:
            st.error(f"Colonnes manquantes dans la feuille 'Historical temp & HDD': "
                     f"TEMP {missing_t} | HDD {missing_h}")
            st.write("Colonnes disponibles :", list(df_th.columns))
            st.stop()

        # conversions numériques
        for c in list(temp_cols.values()) + list(hdd_cols.values()):
            df_th[c] = pd.to_numeric(df_th[c], errors="coerce")
        df_th = df_th.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        st.caption(f"Dernière date lue : **{df_th['Date'].max().date()}**  (mtime: {int(file_mtime)})")

        # ---------- Sélecteur pays ----------
        country = st.selectbox("Choisir un pays :", list(temp_cols.keys()), index=0)

        # ---------- Préparation: années & ticks mois ----------
        df_tmp = df_th[["Date", temp_cols[country]]].rename(columns={temp_cols[country]: "Temp"}).dropna()
        df_tmp["Year"] = df_tmp["Date"].dt.year
        df_tmp["DOY"]  = df_tmp["Date"].dt.dayofyear

        ticks = [pd.Timestamp(2021, m, 1).dayofyear for m in range(1, 13)]
        labels = [calendar.month_abbr[m] for m in range(1, 13)]

        # ---------- FIG 1 : Seasonal Températures ----------
        colors = {
            2025: ("black", 3.0, 1.0),
            2024: ("red",   2.6, 1.0),
            2023: ("green", 2.4, 1.0),
            2022: ("#88c",  1.5, 0.35),
            2021: ("#cc9",  1.5, 0.35),
            2020: ("#9cc",  1.5, 0.35),
        }

        fig_seasonal = go.Figure()
        for yr in sorted(df_tmp["Year"].unique()):
            ys = (
                df_tmp[df_tmp["Year"] == yr]
                .sort_values("DOY")
                .set_index("DOY")["Temp"]
                .rolling(7, min_periods=1).mean()
                .reset_index()
            )
            color, width, opacity = colors.get(yr, ("#bbb", 1.2, 0.3))
            fig_seasonal.add_trace(go.Scatter(
                x=ys["DOY"], y=ys["Temp"], mode="lines",
                name=str(yr), line=dict(color=color, width=width), opacity=opacity
            ))

        fig_seasonal.update_layout(
            title=f"{country} – Seasonal Temperatures (2020–2025)",
            xaxis=dict(title="Month", tickmode="array", tickvals=ticks, ticktext=labels),
            yaxis_title="Temperature (°C)",
            legend=dict(orientation="h"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=450
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

        # ---------- FIG 2 : HDD Mensuels 2025 vs Moyenne 2020–2024 ----------
        d = df_th[["Date", hdd_cols[country]]].rename(columns={hdd_cols[country]: "HDD"}).dropna()
        d["Year"] = d["Date"].dt.year
        d["Month"] = d["Date"].dt.month
        monthly = d.groupby(["Year", "Month"])["HDD"].sum().unstack(0)

        for yr in range(2020, 2026):
            if yr not in monthly.columns:
                monthly[yr] = 0.0
        monthly = monthly.sort_index(axis=1)

        avg_2020_2024 = monthly.loc[:, 2020:2024].mean(axis=1)
        hdd_2025      = monthly[2025]

        x = list(range(1, 13))
        month_lbls = [calendar.month_abbr[m] for m in x]
        fig_hdd = go.Figure()
        width = 0.35
        fig_hdd.add_trace(go.Bar(
            x=[xi - width/2 for xi in x], y=avg_2020_2024.values,
            name=f"{country} Avg 2020–2024", marker_color="black", width=width
        ))
        fig_hdd.add_trace(go.Bar(
            x=[xi + width/2 for xi in x], y=hdd_2025.values,
            name=f"{country} 2025", marker_color="red", width=width
        ))
        fig_hdd.update_layout(
            title=f"{country} – Monthly HDD: 2025 vs Avg",
            xaxis=dict(title="Month", tickmode="array", tickvals=x, ticktext=month_lbls),
            yaxis_title="Number of HDD Days",
            barmode="group",
            legend=dict(orientation="h"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=420
        )
        st.plotly_chart(fig_hdd, use_container_width=True)

    except Exception as e:
        st.error("Une erreur a empêché l’affichage de l’onglet.")
        st.exception(e)
