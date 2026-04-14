import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- EINSTELLUNGEN ---
lb_corr = 30
z_thresh_high = 1
z_thresh_low = -1
vix_spread_trigger = -4 

def get_all_data():
    tickers = ['^GSPC', 'XLK', 'XLI', 'XLY', 'XLF', 'XLV', 'QQQ', 'GLD', 'BTC-USD', '^VIX', '^VIX3M']
    print("Lade Marktdaten...")
    data = yf.download(tickers, start="2005-01-01", progress=False, auto_adjust=False)['Close'].dropna()
    return data

# --- DATEN & LOGIK ---
data = get_all_data()
returns = data.pct_change()
sectors = ['XLK', 'XLI', 'XLY', 'XLF', 'XLV']
corr_series = returns[sectors].rolling(window=lb_corr).corr(returns['^GSPC']).mean(axis=1).dropna()

df_pdv = pd.DataFrame({'SPX': data['^GSPC'], 'VIX': data['^VIX']}).dropna()
df_pdv['Return'] = np.log(df_pdv['SPX'] / df_pdv['SPX'].shift(1))
dt = 1/252

def compute_f(values, l0, l1, theta):
    f0, f1 = 0.0, 0.0
    res = []
    for v in values:
        f0 = f0 * np.exp(-l0 * dt) + v * l0 * dt
        f1 = f1 * np.exp(-l1 * dt) + v * l1 * dt
        res.append((1 - theta) * f0 + theta * f1)
    return np.array(res)

r1 = compute_f(df_pdv['Return'].fillna(0).values, 52.8, 3.79, 0.81)
r2 = compute_f((df_pdv['Return']**2).fillna(0).values, 17.3, 1.16, 0.43)
df_pdv['VIX_PDV'] = (0.054 + (-0.078) * (r1 * 252) + 0.82 * (np.sqrt(r2) * np.sqrt(252))) * 100
df_pdv['Spread'] = df_pdv['VIX'] - df_pdv['VIX_PDV']
raw_z = (df_pdv['VIX_PDV'] - df_pdv['VIX_PDV'].rolling(252).mean()) / df_pdv['VIX_PDV'].rolling(252).std()
df_pdv['Z_Score'] = raw_z.rolling(5).mean()

vix_ratio = (data['^VIX3M'] / data['^VIX']).dropna()
v_ema = vix_ratio.ewm(span=14, adjust=False).mean()

# --- FARB-LOGIK (STATE MACHINE) ---
color_state, current_override = [], False
for i in range(len(df_pdv)):
    z, spread = df_pdv['Z_Score'].iloc[i], df_pdv['Spread'].iloc[i]
    if z < z_thresh_high: current_override = False
    if spread < vix_spread_trigger: current_override = True
    if z > z_thresh_high and not current_override: color_state.append(1)
    elif z < z_thresh_low: color_state.append(-1)
    else: color_state.append(0)
df_pdv['color_state'] = color_state

# --- PLOTTING ---
specs = [[{"secondary_y": True}], [{}], [{}], [{}], [{}]]
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.015, 
                    row_heights=[0.3, 0.175, 0.175, 0.175, 0.175], specs=specs)

# 1. S&P 500 (Primary Y - Log)
ath = data['^GSPC'].cummax()
fig.add_trace(go.Scatter(x=data.index, y=ath, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['^GSPC'], name="S&P 500", line=dict(color="black", width=2), 
                         fill='tonexty', fillcolor='rgba(150, 150, 150, 0.2)'), row=1, col=1)

# --- OVERLAY INDIKATOREN (Secondary Y - Linear) ---
# Diese Traces nutzen die sekundäre Achse und sind initial unsichtbar
fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series, name="O_Korr", 
                         line=dict(color="red", width=1.3), visible=False, opacity=0.7), secondary_y=True, row=1, col=1)
fig.add_trace(go.Scatter(x=v_ema.index, y=v_ema, name="O_Ratio", 
                         line=dict(color="orange", width=1.3), visible=False, opacity=0.7), secondary_y=True, row=1, col=1)
fig.add_trace(go.Scatter(x=df_pdv.index, y=df_pdv['Z_Score'], name="O_Zscore", 
                         line=dict(color="blue", width=1.3), visible=False, opacity=0.7), secondary_y=True, row=1, col=1)
fig.add_trace(go.Scatter(x=df_pdv.index, y=df_pdv['Spread'], name="O_Spread", 
                         line=dict(color="purple", width=1.3), visible=False, opacity=0.7), secondary_y=True, row=1, col=1)

# Hintergrundfarben
df_colors = df_pdv.dropna(subset=['color_state']).copy()
df_colors['change'] = df_colors['color_state'].ne(df_colors['color_state'].shift())
change_indices = df_colors.index[df_colors['change']].tolist() + [df_colors.index[-1]]
for i in range(len(change_indices)-1):
    start, end = change_indices[i], change_indices[i+1]
    state = df_colors.loc[start, 'color_state']
    if state != 0:
        c = "rgba(255,0,0,0.12)" if state == 1 else "rgba(0,255,0,0.12)"
        fig.add_vrect(x0=start, x1=end, fillcolor=c, layer="below", line_width=0, row=1, col=1)

# Signale (Pfeile)
signals = df_pdv[df_pdv['Spread'] < vix_spread_trigger]
fig.add_trace(go.Scatter(x=signals.index, y=signals['SPX'], mode='markers', 
                         marker=dict(symbol='triangle-up', size=10, color='green', line=dict(width=1, color='white'))), row=1, col=1)

# 2-5. Die Indikatoren Charts (Einzeln)
def add_ind(fig, x, y, name, color, row, hline=None, hcolor="red"):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=color, width=1.5)), row=row, col=1)
    if hline is not None: fig.add_hline(y=hline, line_dash="dash", line_color=hcolor, row=row, col=1)
    m, s = y.mean(), y.std()
    for v, l in {m: "M", m+s: "+1", m-s: "-1", m+2*s: "+2", m-2*s: "-2"}.items():
        fig.add_hline(y=v, line_dash="dot", line_color="rgba(128,128,128,0.2)", line_width=1, row=row, col=1)

add_ind(fig, corr_series.index, corr_series, "Korr", "red", 2, 0.9)
add_ind(fig, v_ema.index, v_ema, "Ratio", "orange", 3, 1.0, "blue")
add_ind(fig, df_pdv.index, df_pdv['Z_Score'], "Z-Score", "blue", 4, 1.0)
fig.add_hline(y=-1, line_dash="dash", line_color="green", row=4, col=1)
add_ind(fig, df_pdv.index, df_pdv['Spread'], "Spread", "purple", 5, vix_spread_trigger, "green")

# --- MENÜ-BUTTONS ---
# Wir steuern die Visibility der ersten 6 Traces im Hauptchart (S&P, ATH, Korr_O, Ratio_O, Z_O, Spr_O)
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons", direction="right", active=0, x=0.01, y=1.12,
            buttons=list([
                dict(label="Kein Overlay", method="update", args=[{"visible": [True, True, False, False, False, False] + [True]*25}]),
                dict(label="+ Korr", method="update", args=[{"visible": [True, True, True, False, False, False] + [True]*25}]),
                dict(label="+ Z-Score", method="update", args=[{"visible": [True, True, False, False, True, False] + [True]*25}]),
                dict(label="+ Spread", method="update", args=[{"visible": [True, True, False, False, False, True] + [True]*25}]),
                dict(label="Alle Overlays", method="update", args=[{"visible": [True, True, True, True, True, True] + [True]*25}]),
            ]),
        )
    ]
)

# --- KONFIGURATION DER ACHSEN ---

# Primäre Y-Achsen (S&P auf Log)
fig.update_yaxes(type="log", row=1, col=1)

# Sekundäre Y-Achse (Overlay auf Linear)
fig.update_yaxes(
    secondary_y=True, 
    type="linear", 
    autorange=True, 
    fixedrange=False, 
    title_text="<b>Overlay Skala</b>", 
    row=1, col=1
)

# Alle anderen Achsen auf Autoscale & Linear
for i in range(2, 6):
    fig.update_yaxes(type="linear", autorange=True, fixedrange=False, row=i, col=1)

# Layout & X-Achse
fig.update_layout(height=1050, template="plotly_white", margin=dict(l=60, r=60, t=130, b=50), 
                  hovermode="x unified", uirevision='constant', showlegend=False)

fig.update_xaxes(matches='x', showspikes=True, spikethickness=1, gridcolor='rgba(230,230,230,0.5)')
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.02), row=5, col=1)

# Titel für die Y-Achsen links
labels = ["S&P 500", "CORR", "VIX RATIO", "Z-SCORE", "SPREAD"]
for i, lbl in enumerate(labels, 1): fig.update_yaxes(title_text=f"<b>{lbl}</b>", row=i, col=1)

fig.show()
