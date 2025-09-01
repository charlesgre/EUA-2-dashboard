import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import os
import io, hashlib

# Optionnel : AG Grid pour filtres/tri type Excel
HAS_AGGRID = False
ColumnsAutoSizeMode = None  # par défaut (certaines versions ne l'exposent pas)

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    try:
        # Présent dans certaines versions seulement
        from st_aggrid import ColumnsAutoSizeMode  # type: ignore
    except Exception:
        ColumnsAutoSizeMode = None
    HAS_AGGRID = True
except Exception as e:
    HAS_AGGRID = False
    AGGRID_IMPORT_ERROR = str(e)  # utile pour debug si besoin


st.set_page_config(page_title="Gas Dashboard", layout="wide")
st.title("\U0001F4CA EUA Analytics Dashboard")

st.caption("RUN MARKER: vTempHDD-001")
from pathlib import Path; import os
st.caption("Script path: " + str(Path(__file__).resolve()))
st.caption("CWD: " + os.getcwd())
st.caption("Tabs count (debug): " + str(len(st.tabs if 'tabs' in globals() else [])))

# --- rendre la barre d'onglets scrollable ET autoriser le retour à la ligne ---
st.markdown("""
<style>
/* Autoriser le retour à la ligne si ça déborde */
.stTabs [role="tablist"]{
  display: flex; 
  flex-wrap: wrap !important;
  gap: .25rem .5rem;
}
/* Eviter que les tabs s'étirent */
.stTabs [role="tab"]{
  flex: 0 0 auto !important;
}
/* Légère réduction de padding/texte pour gagner de la place */
.stTabs [role="tab"] > div[data-testid="stMarkdownContainer"]{
  font-size: 0.95rem;
  padding: 0.2rem 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# --- chemins robustes ---
APP_DIR = Path(__file__).resolve().parent
file_path = APP_DIR / "Gas storages.xlsx"   # évite les surprises de CWD
auctions_path = APP_DIR / "Auctions EUA.xlsx"

tabs = st.tabs([
    "Stocks",          # tabs[0]
    "Prix (EUA/TTF)",  # tabs[1]
    "Strats RSI",      # tabs[2]
    "Open Interest",   # tabs[3]
    "Temp/HDD",        # tabs[4]
    "Forward Curve",   # tabs[5]
    "Auctions"         # tabs[6]  ⬅️ nouveau
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

# --- Série pays (typage, nettoyage) ---
series = df_stock[['Date', selected_country]].copy()
series = series.dropna(subset=[selected_country])               # on ne garde que les dates avec valeur
series['Value'] = pd.to_numeric(series[selected_country], errors='coerce')
series = series.dropna(subset=['Value'])
series = series[series['Date'].dt.year >= start_year].copy()

# Info utile : dernière date dispo pour CE pays
last_country_date = series['Date'].max()
st.caption(f"Dernière date disponible ({country_map[selected_country]}) : **{last_country_date.date()}**")

# --- Calendrier 365 jours (on enlève 29/02 et on compresse les jours qui suivent en année bissextile) ---
series['month']   = series['Date'].dt.month
series['day']     = series['Date'].dt.day
series['is_leap'] = series['Date'].dt.is_leap_year

# 1) on enlève le 29/02
series = series[~((series['month'] == 2) & (series['day'] == 29))].copy()

# 2) day-of-year compressé (entier)
series['DOY365'] = series['Date'].dt.dayofyear - (
    ((series['is_leap']) & (series['month'] > 2)).astype(int)
)
series['DOY365'] = series['DOY365'].astype(int)

# ---------- Bande min/max + moyenne 2020–2024 (365j) ----------
range_data = series[series['Date'].dt.year <= 2024].copy()

def year_vector_365(df_year: pd.DataFrame) -> np.ndarray:
    # moyenne par jour, puis réindex 1..365 et interpolation pour lisser les trous
    y = (df_year.groupby('DOY365', as_index=True)['Value'].mean()
         .reindex(np.arange(1, 366))
         .interpolate(limit_direction='both'))
    return y.values

all_years = []
for yr in range(2020, 2025):
    ydf = range_data[range_data['Date'].dt.year == yr]
    all_years.append(year_vector_365(ydf) if not ydf.empty else np.full(365, np.nan))

all_years_array = np.vstack(all_years)
min_vals  = np.nanmin(all_years_array, axis=0)
max_vals  = np.nanmax(all_years_array, axis=0)
mean_vals = np.nanmean(all_years_array, axis=0)

full_doy = np.arange(1, 366)

# Ticks = 1er de chaque mois (année non bissextile)
mois       = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mois_debut = [pd.Timestamp(2021, m, 1).dayofyear for m in range(1, 13)]

# ---------- Figure ----------
fig = go.Figure()

# Bande min-max
fig.add_trace(go.Scatter(
    x=full_doy, y=min_vals, mode='lines', line=dict(color='lightgray'), showlegend=False
))
fig.add_trace(go.Scatter(
    x=full_doy, y=max_vals, mode='lines', fill='tonexty',
    line=dict(color='lightgray'), name='Min-Max 2020–2024',
    fillcolor='rgba(128,128,128,0.3)'
))
# Moyenne
fig.add_trace(go.Scatter(
    x=full_doy, y=mean_vals, mode='lines', name='Moyenne 2020–2024',
    line=dict(color='black', dash='dash')
))

# Courbes par année (triées par DOY, pour éviter les ruptures visuelles)
for yr in range(start_year, end_year + 1):
    ydf = series[series['Date'].dt.year == yr].copy()
    if not ydf.empty:
        ydf = ydf.sort_values('DOY365')                    # ✅ tri
        fig.add_trace(go.Scatter(
            x=ydf['DOY365'], y=ydf['Value'],
            mode='lines', name=str(yr),
            line=dict(width=2 if yr >= 2023 else 1),
            opacity=1.0 if yr >= 2023 else 0.4
        ))

fig.update_layout(
    title=f"{country_map[selected_country]} - Stockage de gaz (TWh)",
    xaxis=dict(
        title="Mois",
        tickmode='array',
        tickvals=mois_debut,
        ticktext=mois,
        range=[1, 365]  # démarre pile au 1er janvier
    ),
    yaxis_title="TWh",
    legend=dict(orientation="h"),
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
with tabs[5]:
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
with tabs[4]:
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

# === 6. Onglet AUCTIONS ===
with tabs[6]:
    st.header("Auctions")

    # --- utilitaires ---
    def _make_unique(names):
        seen, out = {}, []
        for n in names:
            key = "" if n is None else str(n)
            key = key if key.strip() != "" else "Column"
            if key in seen:
                seen[key] += 1
                out.append(f"{key}_{seen[key]}")
            else:
                seen[key] = 0
                out.append(key)
        return out

    def _file_version(p: Path) -> str:
        return f"{int(os.path.getmtime(p))}:{p.stat().st_size}"

    @st.cache_data(show_spinner=False)
    def load_auctions_dataframe(xlsx_path: Path, file_version: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx_path, sheet_name=0, header=0)
        part1 = df.iloc[3:16]    # Excel 5..17
        part2 = df.iloc[53:92]   # Excel 55..93
        out = pd.concat([part1, part2], axis=0).copy()
        # Laisse les NaN en NaN (ils s'afficheront vides via na_rep / AgGrid)
        out = out.replace({"N/A": np.nan, "n/a": np.nan, "#N/A": np.nan})

        # Typage : force les colonnes numériques en float, et la date si présente
        num_candidates = [
            "Last","Vol","Cover Ratio","Mkt Diff","Total Bids (Volume)",
            "Min","Max","Mean","Median","Sel. Bids","Tot Bids"
        ]
        for col in [c for c in num_candidates if c in out.columns]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

        out.columns = _make_unique(out.columns)
        return out

    if not auctions_path.exists():
        st.error(f"Fichier introuvable : {auctions_path}")
        st.stop()

    df_auctions = load_auctions_dataframe(auctions_path, _file_version(auctions_path))
    st.caption("Lignes affichées : **5–17** et **55–93** (entêtes = ligne 1).")

    # --- rendu dynamique (AgGrid si dispo, sinon Styler) ---
    if HAS_AGGRID:
        # CSS header + zébrage
        st.markdown("""
        <style>
        .ag-theme-balham .ag-header{
            background: linear-gradient(180deg,#f7f9fc 0%,#eef3fb 100%);
            font-weight: 600; border-bottom: 1px solid #dbe4f0;
        }
        .ag-theme-balham .ag-header-cell-label { color:#334155; }
        .ag-theme-balham .ag-row:nth-child(even) .ag-cell { background:#fafafa; }
        .ag-theme-balham .ag-cell { line-height:1.15rem; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            quick = st.text_input("🔎 Rechercher (quick filter)", "")
        with c2:
            use_colors = st.checkbox("Couleurs conditionnelles", True)

        # Certaines versions exposent JsCode
        try:
            from st_aggrid import JsCode  # type: ignore
        except Exception:
            JsCode = None

        gb = GridOptionsBuilder.from_dataframe(df_auctions)
        gb.configure_default_column(filter=True, sortable=True, resizable=True, headerClass="bold-header")

        # Formatage FR pour colonnes numériques
        num_candidates = [
            "Last","Vol","Cover Ratio","Mkt Diff","Total Bids (Volume)",
            "Min","Max","Mean","Median","Sel. Bids","Tot Bids"
        ]
        for col in [c for c in num_candidates if c in df_auctions.columns]:
            gb.configure_column(
                col,
                type=["numericColumn","numberColumnFilter","customNumericFormat"],
                valueFormatter="(x==null || x==='') ? '' : Intl.NumberFormat('fr-FR',{maximumFractionDigits:2}).format(x)"
            )

        # Date au format FR
        if "Date" in df_auctions.columns:
            gb.configure_column("Date", valueFormatter="value ? new Date(value).toLocaleDateString('fr-FR') : ''")

        # Gras sur "Last"
        if "Last" in df_auctions.columns and JsCode:
            gb.configure_column("Last", cellStyle=JsCode(
                "function(p){ if(p.value==null) return {}; return {'fontWeight':'700'}; }"
            ))

        # Couleurs conditionnelles sur "Cover Ratio"
        if use_colors and "Cover Ratio" in df_auctions.columns and JsCode:
            gb.configure_column("Cover Ratio", cellStyle=JsCode("""
                function(p){
                    if(p.value==null) return {};
                    if(p.value >= 1.5) return {'background-color':'#e7f6e7','color':'#0a662a','font-weight':'600'};
                    if(p.value <  1.2) return {'background-color':'#fdecea','color':'#b71c1c','font-weight':'600'};
                    return {};
                }
            """))

        grid_opts = gb.build()
        grid_opts["quickFilterText"] = quick
        grid_opts["animateRows"] = True
        grid_opts["rowHeight"] = 32
        grid_opts["pagination"] = True
        grid_opts["paginationPageSize"] = 20

        # kwargs optionnels selon la version
        ag_kwargs = dict(
            gridOptions=grid_opts,
            height=560,
            fit_columns_on_grid_load=True,
        )
        if ColumnsAutoSizeMode is not None:
            ag_kwargs["columns_auto_size_mode"] = ColumnsAutoSizeMode.FIT_CONTENTS

        # Ajouter "theme" seulement si supporté
        try:
            import inspect
            if "theme" in inspect.signature(AgGrid).parameters:
                ag_kwargs["theme"] = "balham"
        except Exception:
            pass

        AgGrid(df_auctions, **ag_kwargs)

    else:
        # -------- Fallback sans AgGrid : Styler sans matplotlib --------
        df_style = df_auctions.copy()

        # Forcer types numériques si présents
        num_candidates = [
            "Last","Vol","Cover Ratio","Mkt Diff","Total Bids (Volume)",
            "Min","Max","Mean","Median","Sel. Bids","Tot Bids"
        ]
        for col in [c for c in num_candidates if c in df_style.columns]:
            df_style[col] = pd.to_numeric(df_style[col], errors="coerce")

        fmt = {c: "{:,.2f}".format for c in df_style.select_dtypes(include=[np.number]).columns}

        def highlight_empty(row):
            # fond blanc pour NaN/vides
            styles = []
            for v in row:
                styles.append("background-color: white;" if pd.isna(v) or (isinstance(v, str) and v.strip() == "") else "")
            return styles

        # Dégradé custom (hex mix) pour Cover Ratio sans matplotlib
        def _mix_hex(c1, c2, t: float) -> str:
            # c1 et c2 comme "#RRGGBB", t in [0,1]
            c1 = c1.lstrip("#"); c2 = c2.lstrip("#")
            r1,g1,b1 = int(c1[0:2],16), int(c1[2:4],16), int(c1[4:6],16)
            r2,g2,b2 = int(c2[0:2],16), int(c2[2:4],16), int(c2[4:6],16)
            r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
            return f"#{r:02x}{g:02x}{b:02x}"

        styled = (
            df_style
            .style
            .format(fmt, na_rep="")  # affiche vide pour NaN
            .set_table_styles([
                {"selector":"th.col_heading","props":[("background","#eef3fb"),("font-weight","bold"),("color","#334155"),("border-bottom","1px solid #dbe4f0")]},
                {"selector":"thead","props":[("border-bottom","1px solid #dbe4f0")]},
            ])
            .apply(highlight_empty, axis=1)
        )

        # Applique un dégradé vert custom à Cover Ratio s'il existe
        if "Cover Ratio" in df_style.columns:
            s = df_style["Cover Ratio"]
            vmin = float(np.nanmin(s.values)) if np.isfinite(s).any() else 0.0
            vmax = float(np.nanmax(s.values)) if np.isfinite(s).any() else 1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0  # évite division par zéro

            # couleurs du dégradé (clair -> foncé)
            c_lo = "#e7f6e7"
            c_hi = "#0a662a"

            def cover_ratio_styles(col: pd.Series):
                styles = []
                for v in col:
                    if pd.isna(v):
                        styles.append("")  # pas de style
                    else:
                        t = (float(v) - vmin) / (vmax - vmin)
                        if t < 0: t = 0.0
                        if t > 1: t = 1.0
                        bg = _mix_hex(c_lo, c_hi, t)
                        styles.append(f"background-color: {bg}; color: #0b3d0b;")
                return styles

            styled = styled.apply(cover_ratio_styles, subset=["Cover Ratio"], axis=0)

        # Affiche le HTML stylé (st.dataframe ignore souvent le CSS du Styler)
        st.markdown(styled.to_html(), unsafe_allow_html=True)



    # --- exports (toujours affichés) ---
    csv_bytes = df_auctions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger (CSV)",
        data=csv_bytes,
        file_name="auctions_filtre.csv",
        mime="text/csv",
        use_container_width=True,
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_auctions.to_excel(writer, index=False, sheet_name="auctions")
    st.download_button(
        "Télécharger (Excel)",
        data=buf.getvalue(),
        file_name="auctions_filtre.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if st.button("🔄 Rafraîchir les données (Auctions)"):
        load_auctions_dataframe.clear()
        st.rerun()
