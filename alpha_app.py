import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Seite konfigurieren
st.set_page_config(layout="wide", page_title="S&P 500 Alpha")

@st.cache_data(ttl=3600) # Speichert Daten für 1 Stunde
def load_data():
    tickers = ['^GSPC', '^VIX', '^VIX3M', 'XLK', 'XLI', 'XLY', 'XLF', 'XLV']
    # Wir laden ab 2000 für eine solide Historie
    df = yf.download(tickers, start="2018-01-01", progress=False, auto_adjust=True)['Close']
    return df.dropna()

try:
    data = load_data()

    # --- BERECHNUNGEN ---
    # Drawdown
    spx = data['^GSPC']
    rolling_max = spx.cummax()
    drawdown = spx - rolling_max

    # Indikatoren
    returns = data.pct_change()
    sectors = ['XLK', 'XLI', 'XLY', 'XLF', 'XLV']
    corr = returns[sectors].rolling(30).corr(returns['^GSPC']).mean(axis=1)
    vix_ratio = data['^VIX3M'] / data['^VIX']
    
    # Z-Score des Ratio (statistische Abweichung)
    z_score = (vix_ratio - vix_ratio.rolling(252).mean()) / vix_ratio.rolling(252).std()
    
    # VIX Spread
    v_spread = data['^VIX'] - data['^VIX3M']

    # --- DEINE ALPHA LOGIK ---
    # BUY: Korr > 0.9 (1pt), Ratio < 0.9 (1pt), Spread < -2 (1pt) -> Max +3
    s_buy = (corr > 0.9).astype(int) + (vix_ratio < 0.9).astype(int) + (v_spread < -2).astype(int)
    # HEDGE: Z-Score > 1.5 (1pt), Ratio > 1.2 (1pt) -> Max -2
    s_hedge = (z_score > 1.5).astype(int) + (vix_ratio > 1.2).astype(int)
    alpha = s_buy - s_hedge

    # --- PLOTTING ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                        subplot_titles=("S&P 500: DRAWDOWN & ALPHA CONFLUENCE", "CORRELATION", "VIX RATIO", "Z-SCORE", "VIX SPREAD"))

    # Hauptchart mit Drawdown (Grau)
    fig.add_trace(go.Scatter(x=spx.index, y=rolling_max, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=spx.index, y=spx, fill='tonexty', fillcolor='rgba(200,200,200,0.3)', 
                             line=dict(color='black', width=2), name="S&P 500"), row=1, col=1)

    # Alpha Shading (Die "Tapetenstreifen")
    # Wir filtern nur die Tage, an denen Alpha nicht 0 ist, um Speed zu gewinnen
    active_alpha = alpha[alpha != 0]
    for val, color in [(1,"rgba(0,255,0,0.1)"), (2,"rgba(0,255,0,0.2)"), (3,"rgba(0,255,0,0.3)"),
                       (-1,"rgba(255,0,0,0.1)"), (-2,"rgba(255,0,0,0.2)")]:
        days = active_alpha[active_alpha == val].index
        if len(days) > 0:
            for day in days[::3]: # Performance: Jeden 3. Tag zeichnen reicht für die Optik
                fig.add_vrect(x0=day, x1=day, fillcolor=color, line_width=0, layer="below", row=1, col=1)

    # Untere Charts
    fig.add_trace(go.Scatter(x=corr.index, y=corr, name="Korr", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=vix_ratio.index, y=vix_ratio, name="Ratio", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name="Z-Score", line=dict(color="purple")), row=4, col=1)
    fig.add_trace(go.Scatter(x=v_spread.index, y=v_spread, name="Spread", fill='tozeroy', line=dict(color="green")), row=5, col=1)

    # Schwellenwerte
    fig.add_hline(y=0.9, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=1.2, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=1.5, line_dash="dot", line_color="red", row=4, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=5, col=1)

    fig.update_layout(height=1000, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Daten konnten nicht geladen werden: {e}")
