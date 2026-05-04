"""
Signal Generator V11 HMM — GitHub Actions  [VIX MA12]
======================================================
Sincronizado con: analyze_period_V11_HMM_empirical_varios_VIX_filters.py

Lógica:
  - Descarga datos desde OOS_START (2023-01-01)
  - Clasifica régimen con HMM Viterbi (bulk desde OOS_START)
  - Aplica filtro VIX MA12: si VIX_close_D > MA12 → no operar
  - Si pasa el filtro y prob >= threshold → envía webhook SPS
  - Sin cooldown (se maneja por fuera con datos reales)
  - Sin pricing (no necesario para la señal)
  - Sin adaptive threshold, sin dynamic sizing

Filtro VIX MA12 (sin lookahead):
  - Usa VIX close del día de análisis D
  - Disponible antes del open D+1 (día de entrada)

CSV generado (señales desde OOS_START):
  date, signal_date, regime, regime_disabled,
  vix_close, vix_ma12, vix_filtered,
  prob_bullish, threshold, passed_threshold,
  signal_type, signal_generated

Variable de entorno requerida en GitHub Actions:
  OA_WEBHOOK_SPS

USAGE:
  python signal_generator_V11_HMM_empirical.py
  (misma carpeta que models_V11_hmm/)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import pickle
import requests
import json

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_V11_hmm")

if not os.path.exists(MODELS_DIR):
    parent    = os.path.dirname(BASE_DIR)
    candidate = os.path.join(parent, "models_V11_hmm")
    if os.path.exists(candidate):
        BASE_DIR   = parent
        MODELS_DIR = candidate

OOS_START = "2023-01-01"

# =============================================================================
# CONFIGURATION
# =============================================================================
SIGNAL_CONFIG = {
    "webhook_sps": os.environ.get("OA_WEBHOOK_SPS"),

    "vix_ma_cross": {
        "enabled":   True,
        "ma_period": 12,
    },

    "regime_multipliers": {
        "LOW_VOL_BULL":    0,
        "HIGH_VOL_BEAR_6": 0,
    },

    "trade_sps": True,
    "trade_scs": False,
    "generate_csv": True,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def normalize(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col).lower() for col in df.columns]
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if "open"  in cl:                        rename[c] = "Open"
        if "high"  in cl:                        rename[c] = "High"
        if "low"   in cl and "close" not in cl:  rename[c] = "Low"
        if "close" in cl:                        rename[c] = "Close"
    df = df.rename(columns=rename)
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])


def add_features(df):
    df["ret_1d"]  = df["Close"].pct_change()
    df["ret_2d"]  = df["Close"].pct_change(2)
    df["ret_3d"]  = df["Close"].pct_change(3)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_5d"]  = df["ret_1d"].rolling(5).std()
    df["SMA8"]    = df["Close"].rolling(8).mean()
    df["SMA9"]    = df["Close"].rolling(9).mean()
    df["SMA20"]   = df["Close"].rolling(20).mean()
    df["SMA50"]   = df["Close"].rolling(50).mean()
    df["price_above_sma8"]  = (df["Close"] > df["SMA8"]).astype(int)
    df["price_above_sma9"]  = (df["Close"] > df["SMA9"]).astype(int)
    df["price_above_sma20"] = (df["Close"] > df["SMA20"]).astype(int)
    df["price_sma9"]        = df["Close"] / df["SMA9"]
    df["price_sma20"]       = df["Close"] / df["SMA20"]

    period = 14
    df["high_diff"] = df["High"].diff()
    df["low_diff"]  = -df["Low"].diff()
    df["plus_dm"]   = np.where((df["high_diff"] > df["low_diff"]) & (df["high_diff"] > 0), df["high_diff"], 0)
    df["minus_dm"]  = np.where((df["low_diff"] > df["high_diff"]) & (df["low_diff"] > 0),  df["low_diff"],  0)
    df["tr"]        = np.maximum(df["High"] - df["Low"],
                       np.maximum(abs(df["High"] - df["Close"].shift(1)),
                                  abs(df["Low"]  - df["Close"].shift(1))))
    df["atr"]       = df["tr"].rolling(period).mean()
    df["plus_di"]   = 100 * (df["plus_dm"].rolling(period).mean() / df["atr"])
    df["minus_di"]  = 100 * (df["minus_dm"].rolling(period).mean() / df["atr"])
    df["dx"]        = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
    df["adx"]       = df["dx"].rolling(period).mean()

    delta   = df["Close"].diff()
    g21     = (delta.where(delta > 0, 0)).rolling(21).mean()
    l21     = (-delta.where(delta < 0, 0)).rolling(21).mean()
    df["rsi_21"] = 100 - (100 / (1 + g21 / l21))
    g14     = (delta.where(delta > 0, 0)).rolling(14).mean()
    l14     = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + g14 / l14))

    if "VIX" in df.columns:
        df["vix"]        = df["VIX"]
        df["vix_chg_5d"] = df["VIX"].pct_change(5)
    df["trend_score"] = df["SMA20"] / df["SMA50"]

    df = df.drop(["high_diff", "low_diff", "plus_dm", "minus_dm", "tr", "dx"],
                 axis=1, errors="ignore")
    return df


# =============================================================================
# HMM
# =============================================================================
_hmm_model = _hmm_scaler = _hmm_features = None

def load_hmm():
    global _hmm_model, _hmm_scaler, _hmm_features
    if _hmm_model is not None:
        return _hmm_model, _hmm_scaler, _hmm_features
    with open(os.path.join(MODELS_DIR, "hmm_model.pkl"),     "rb") as f: _hmm_model    = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "hmm_scaler.pkl"),    "rb") as f: _hmm_scaler   = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "hmm_features.json"), "r")  as f: _hmm_features = json.load(f)["features"]
    return _hmm_model, _hmm_scaler, _hmm_features


def classify_regimes_bulk(df, regime_config):
    hmm_model, scaler, features = load_hmm()
    df_lower = df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]
    df_lower = df_lower.loc[:, ~df_lower.columns.duplicated()]
    if "vix" in df_lower.columns and "vix_chg_5d" not in df_lower.columns:
        df_lower["vix_chg_5d"] = df_lower["vix"].pct_change(5)
    if "sma20" in df_lower.columns and "sma50" in df_lower.columns \
            and "trend_score" not in df_lower.columns:
        df_lower["trend_score"] = df_lower["sma20"] / df_lower["sma50"]
    missing = [f for f in features if f not in df_lower.columns]
    if missing:
        print(f"  WARNING HMM features faltantes: {missing}")
    valid  = df_lower[features].dropna()
    result = pd.Series(index=df.index, dtype=object)
    if len(valid) > 0:
        X_scaled    = scaler.transform(valid[features].to_numpy())
        states      = hmm_model.predict(X_scaled)
        id_to_label = {p["hmm_id"]: name for name, p in regime_config["regimes"].items()}
        for idx, state in zip(valid.index, states):
            result[idx] = id_to_label.get(int(state), f"REGIME_{state}")
    result = result.fillna(next(iter(regime_config["regimes"])))
    return result


# =============================================================================
# LOAD MODELS
# =============================================================================
def load_models():
    config_file = os.path.join(MODELS_DIR, "regime_config.json")
    if not os.path.exists(config_file):
        print(f"  No models found at {MODELS_DIR}")
        return None, None
    with open(config_file, "r") as f:
        regime_config = json.load(f)
    models = {}
    for regime_name in regime_config["regimes"].keys():
        model_file = os.path.join(MODELS_DIR, f"regime_{regime_name}_model.pkl")
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                models[regime_name] = pickle.load(f)
    return models, regime_config


# =============================================================================
# WEBHOOK
# =============================================================================
def send_webhook(signal_type):
    url = SIGNAL_CONFIG.get(f"webhook_{signal_type.lower()}")
    if not url:
        print(f"  WARNING: OA_WEBHOOK_{signal_type.upper()} no configurado — señal NO enviada")
        return False
    try:
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            print(f"  OK Webhook enviado: {signal_type}")
            return True
        else:
            print(f"  ERROR Webhook HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ERROR Webhook: {e}")
        return False


# =============================================================================
# CSV HISTORIAL OOS
# =============================================================================
def generate_oos_csv(spx, models, regime_config, config):
    print(f"\n  Generando CSV desde {OOS_START}...")
    try:
        regime_mults = config.get("regime_multipliers", {})
        trade_sps    = config.get("trade_sps", True)
        trade_scs    = config.get("trade_scs", False)
        vix_ma_cfg   = config.get("vix_ma_cross", {})
        ma_enabled   = vix_ma_cfg.get("enabled", True)
        ma_period    = vix_ma_cfg.get("ma_period", 12)

        cutoff = pd.Timestamp(OOS_START)

        # Viterbi bulk — sobre todo el spx descargado
        all_regimes = classify_regimes_bulk(spx, regime_config)

        # VIX MA sin shift — usa close de D directamente
        vix_ma_series = spx["VIX"].rolling(ma_period).mean() \
                        if ("VIX" in spx.columns and ma_enabled) \
                        else pd.Series(np.nan, index=spx.index)

        spx_oos = spx[spx.index >= cutoff]
        rows    = []
        n       = len(spx_oos)

        for i in range(n - 2):
            sig_day     = spx_oos.iloc[i]
            signal_date = spx_oos.index[i]
            trade_date  = spx_oos.index[i + 1]

            vix_close = float(spx["VIX"][signal_date]) \
                        if (signal_date in spx.index and "VIX" in spx.columns) else np.nan
            vix_ma_v  = float(vix_ma_series[signal_date]) \
                        if signal_date in vix_ma_series.index else np.nan

            vix_filtered = (ma_enabled
                            and pd.notna(vix_close)
                            and pd.notna(vix_ma_v)
                            and vix_close > vix_ma_v)

            regime          = all_regimes.get(signal_date, None)
            mult            = regime_mults.get(regime, 1) if regime else 0
            regime_disabled = (mult == 0)

            prob_bullish     = None
            prob_bearish     = None
            passed_threshold = False
            signal_generated = False
            signal_type_out  = None
            threshold_used   = None

            if not vix_filtered and not regime_disabled and regime in models:
                model  = models[regime]
                params = regime_config["regimes"][regime]
                X = pd.DataFrame([{
                    "ret_2d":            float(sig_day.get("ret_2d",            0)),
                    "price_above_sma8":  float(sig_day.get("price_above_sma8",  0)),
                    "ret_20d":           float(sig_day.get("ret_20d",           0)),
                    "price_above_sma9":  float(sig_day.get("price_above_sma9",  0)),
                    "ret_1d":            float(sig_day.get("ret_1d",            0)),
                    "price_above_sma20": float(sig_day.get("price_above_sma20", 0)),
                    "adx":               float(sig_day.get("adx",               25)),
                    "rsi_21":            float(sig_day.get("rsi_21",            50)),
                    "price_sma20":       float(sig_day.get("price_sma20",       1.0)),
                    "minus_di":          float(sig_day.get("minus_di",          25)),
                    "vol_5d":            float(sig_day.get("vol_5d",            0.01)),
                }])
                try:
                    prob_bullish = float(model.predict_proba(X)[0, 1])
                    prob_bearish = 1.0 - prob_bullish
                except Exception:
                    pass

                if prob_bullish is not None:
                    candidates = []
                    if trade_sps: candidates.append(("SPS", prob_bullish, float(params.get("threshold_sps", 0.5))))
                    if trade_scs: candidates.append(("SCS", prob_bearish, float(params.get("threshold_scs", 0.6))))
                    for s_type, s_prob, s_base in candidates:
                        threshold_used = s_base
                        if s_prob >= s_base:
                            signal_generated = True
                            passed_threshold = True
                            signal_type_out  = s_type
                            break

            rows.append({
                "date":             trade_date.strftime("%Y-%m-%d"),
                "signal_date":      signal_date.strftime("%Y-%m-%d"),
                "regime":           regime if regime else "",
                "regime_disabled":  regime_disabled,
                "vix_close":        round(vix_close, 4) if pd.notna(vix_close) else None,
                "vix_ma12":         round(vix_ma_v,  4) if pd.notna(vix_ma_v)  else None,
                "vix_filtered":     vix_filtered,
                "prob_bullish":     round(prob_bullish, 4) if prob_bullish is not None else None,
                "threshold":        threshold_used,
                "passed_threshold": passed_threshold,
                "signal_type":      signal_type_out if signal_type_out else "",
                "signal_generated": signal_generated,
            })

        if rows:
            csv_df    = pd.DataFrame(rows)
            today_str = datetime.now().strftime("%Y%m%d")
            start_str = cutoff.strftime("%Y%m%d")
            csv_path  = os.path.join(BASE_DIR, f"signals_{start_str}_{today_str}.csv")
            csv_df.to_csv(csv_path, index=False)
            n_sig = csv_df["signal_generated"].sum()
            n_blk = csv_df["vix_filtered"].sum()
            print(f"  CSV guardado: {csv_path}")
            print(f"  Total dias: {len(csv_df)}  |  Senales: {n_sig}  |  Bloqueadas VIX MA: {n_blk}")
        else:
            print(f"  Sin datos desde {OOS_START}")

    except Exception as e:
        print(f"  Error generando CSV: {e}")
        import traceback; traceback.print_exc()


# =============================================================================
# SUMMARY
# =============================================================================
def print_summary(summary):
    ma_period = SIGNAL_CONFIG.get("vix_ma_cross", {}).get("ma_period", 12)
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Fecha:             {summary['date']}")
    print(f"SPX close:         ${summary['spx']:.2f}"      if summary["spx"]              else "SPX:               N/A")
    print(f"VIX close:         {summary['vix']:.2f}"       if summary["vix"] is not None  else "VIX:               N/A")
    print(f"VIX MA{ma_period}:          {summary['vix_ma']:.2f}"  if summary.get("vix_ma") is not None else f"VIX MA{ma_period}:          N/A")
    print(f"VIX MA filter:     {'BLOQUEADO' if summary.get('vix_filtered') else 'OK'}")
    print(f"Regimen:           {summary['regime']}"        if summary["regime"]           else "Regimen:           N/A")
    print("-" * 40)
    if summary.get("prob_bullish") is not None:
        print(f"Prob Bullish:      {summary['prob_bullish']*100:.1f}%")
    if summary.get("threshold_sps") is not None:
        print(f"Threshold SPS:     {summary['threshold_sps']*100:.1f}%")
    print(f"Signal:            {'SPS - ENVIADA' if summary['signal_generated'] else 'NO'}")
    print(f"Motivo:            {summary['reason']}")
    print("=" * 65)


# =============================================================================
# GENERATE SIGNAL
# =============================================================================
def generate_signal():
    print("=" * 65)
    print("SIGNAL GENERATOR V11 HMM [VIX MA12]")
    print("=" * 65)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = SIGNAL_CONFIG

    summary = {
        "date":             datetime.now().strftime("%Y-%m-%d"),
        "spx":              None,
        "vix":              None,
        "vix_ma":           None,
        "vix_filtered":     False,
        "regime":           None,
        "prob_bullish":     None,
        "threshold_sps":    None,
        "signal_generated": False,
        "reason":           None,
    }

    # Modelos
    print(f"\nCargando modelos...")
    models, regime_config = load_models()
    if not models:
        summary["reason"] = "No models found"
        print_summary(summary); return
    print(f"  {len(models)} modelos cargados")

    # Datos con warmup de 400 días antes de OOS_START (igual que el analyzer)
    # para que el Viterbi HMM tenga exactamente la misma historia
    print(f"\nDescargando datos (400d warmup pre-{OOS_START})...")
    now = datetime.now()
    oos_dt       = datetime.strptime(OOS_START, "%Y-%m-%d")
    download_start = oos_dt - pd.Timedelta(days=400)
    try:
        spx = normalize(yf.download("^GSPC", start=download_start.strftime("%Y-%m-%d"),
                                    end=now.strftime("%Y-%m-%d"), progress=False))
        vix = normalize(yf.download("^VIX",  start=download_start.strftime("%Y-%m-%d"),
                                    end=now.strftime("%Y-%m-%d"), progress=False))
        if spx is None or len(spx) == 0:
            summary["reason"] = "SPX download failed"; print_summary(summary); return
        if vix is None or len(vix) == 0:
            summary["reason"] = "VIX download failed"; print_summary(summary); return
        spx["VIX"] = vix["Close"]
        spx = add_features(spx)
        print(f"  {len(spx)} dias descargados")
    except Exception as e:
        summary["reason"] = f"Download error: {e}"; print_summary(summary); return

    analysis_day  = spx.iloc[-1]
    analysis_date = spx.index[-1]
    spx_close     = float(analysis_day["Close"])
    vix_close     = float(analysis_day.get("VIX", 0))
    summary["spx"] = spx_close
    summary["vix"] = vix_close

    print(f"\nDia de analisis: {analysis_date.strftime('%Y-%m-%d')}")
    print(f"  SPX close: ${spx_close:.2f}")
    print(f"  VIX close: {vix_close:.2f}")

    # Cross-validation
    TOLERANCE_PCT = 0.001
    cross_ok      = True
    print(f"\nCross-validando datos...")
    try:
        spx2 = normalize(yf.download("^GSPC", period="30d", progress=False))
        if spx2 is not None and not spx2.empty:
            common    = spx["Close"].index.intersection(spx2["Close"].index)
            diffs     = ((spx["Close"][common] - spx2["Close"][common]).abs() / spx2["Close"][common])
            last_diff = float(diffs[common[-1]]) if common[-1] == analysis_date else None
            print(f"  SPX dirty days: {(diffs > TOLERANCE_PCT).sum()}  max diff: {diffs.max()*100:.3f}%")
            if last_diff is not None and last_diff > TOLERANCE_PCT:
                summary["reason"] = f"SPX ABORT: ultimo dia diff {last_diff*100:.3f}%"
                cross_ok = False
            else:
                print(f"  SPX OK")
    except Exception as e:
        print(f"  SPX secondary: error ({e}) - omitido")

    try:
        cboe_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
        cboe_df  = pd.read_csv(cboe_url, parse_dates=["DATE"]).sort_values("DATE").set_index("DATE")
        vix_yf   = spx["VIX"].dropna()
        common   = vix_yf.index.intersection(cboe_df.index)
        if len(common) > 0:
            cboe_vix  = cboe_df.loc[common, "CLOSE"]
            diffs     = ((vix_yf[common] - cboe_vix).abs() / cboe_vix)
            last_diff = float(diffs[common[-1]]) if common[-1] == analysis_date else None
            print(f"  VIX dirty days: {(diffs > TOLERANCE_PCT).sum()}  max diff: {diffs.max()*100:.3f}%")
            if last_diff is not None and last_diff > TOLERANCE_PCT:
                summary["reason"] = f"VIX ABORT: ultimo dia diff {last_diff*100:.3f}%"
                cross_ok = False
            else:
                print(f"  VIX OK")
    except Exception as e:
        print(f"  VIX CBOE: error ({e}) - omitido")

    if not cross_ok:
        if config.get("generate_csv", True):
            generate_oos_csv(spx, models, regime_config, config)
        print_summary(summary); return

    # Filtro VIX MA12 — usa close de D sin shift (causal, sin lookahead)
    vix_ma_cfg = config.get("vix_ma_cross", {})
    ma_enabled = vix_ma_cfg.get("enabled", True)
    ma_period  = vix_ma_cfg.get("ma_period", 12)

    if ma_enabled:
        vix_ma_series = spx["VIX"].rolling(ma_period).mean()
        vix_ma_val    = float(vix_ma_series[analysis_date]) \
                        if analysis_date in vix_ma_series.index else np.nan
        summary["vix_ma"] = None if np.isnan(vix_ma_val) else vix_ma_val

        print(f"\nFiltro VIX MA{ma_period}:")
        print(f"  VIX close: {vix_close:.2f}")
        print(f"  VIX MA{ma_period}:   {vix_ma_val:.2f}" if not np.isnan(vix_ma_val) else f"  VIX MA{ma_period}:   N/A")

        if not np.isnan(vix_ma_val) and vix_close > vix_ma_val:
            summary["vix_filtered"] = True
            summary["reason"] = f"VIX MA{ma_period} bloqueado: {vix_close:.2f} > {vix_ma_val:.2f}"
            print(f"  -> BLOQUEADO")
            if config.get("generate_csv", True):
                generate_oos_csv(spx, models, regime_config, config)
            print_summary(summary); return
        else:
            print(f"  -> OK")

    # Regimen
    print(f"\nClasificando regimen...")
    all_regimes    = classify_regimes_bulk(spx, regime_config)
    current_regime = all_regimes.get(analysis_date, next(iter(regime_config["regimes"])))
    summary["regime"] = current_regime

    regime_mults = config.get("regime_multipliers", {})
    multiplier   = regime_mults.get(current_regime, 1)
    print(f"  Regimen: {current_regime}  (multiplier={multiplier})")

    if multiplier == 0:
        summary["reason"] = f"Regimen {current_regime} deshabilitado"
        if config.get("generate_csv", True):
            generate_oos_csv(spx, models, regime_config, config)
        print_summary(summary); return

    if current_regime not in models:
        summary["reason"] = f"Sin modelo para {current_regime}"
        if config.get("generate_csv", True):
            generate_oos_csv(spx, models, regime_config, config)
        print_summary(summary); return

    # Prediccion
    model  = models[current_regime]
    params = regime_config["regimes"][current_regime]

    X = pd.DataFrame([{
        "ret_2d":            float(analysis_day.get("ret_2d",            0)),
        "price_above_sma8":  float(analysis_day.get("price_above_sma8",  0)),
        "ret_20d":           float(analysis_day.get("ret_20d",           0)),
        "price_above_sma9":  float(analysis_day.get("price_above_sma9",  0)),
        "ret_1d":            float(analysis_day.get("ret_1d",            0)),
        "price_above_sma20": float(analysis_day.get("price_above_sma20", 0)),
        "adx":               float(analysis_day.get("adx",               25)),
        "rsi_21":            float(analysis_day.get("rsi_21",            50)),
        "price_sma20":       float(analysis_day.get("price_sma20",       1.0)),
        "minus_di":          float(analysis_day.get("minus_di",          25)),
        "vol_5d":            float(analysis_day.get("vol_5d",            0.01)),
    }])

    try:
        prob_bullish = float(model.predict_proba(X)[0, 1])
        prob_bearish = 1.0 - prob_bullish
    except Exception as e:
        summary["reason"] = f"Prediction error: {e}"
        if config.get("generate_csv", True):
            generate_oos_csv(spx, models, regime_config, config)
        print_summary(summary); return

    summary["prob_bullish"] = prob_bullish

    print(f"\nPrediccion:")
    print(f"  prob_bullish (SPS): {prob_bullish*100:.1f}%")

    # Threshold
    trade_sps = config.get("trade_sps", True)
    trade_scs = config.get("trade_scs", False)
    candidates = []
    if trade_sps: candidates.append(("SPS", prob_bullish, float(params.get("threshold_sps", 0.5))))
    if trade_scs: candidates.append(("SCS", prob_bearish, float(params.get("threshold_scs", 0.6))))

    print(f"\nEvaluando senales:")
    signal_type = None
    for s_type, s_prob, s_base in candidates:
        if s_type == "SPS":
            summary["threshold_sps"] = s_base
        print(f"  {s_type}: prob={s_prob*100:.1f}%  threshold={s_base*100:.1f}%")
        if s_prob >= s_base:
            signal_type = s_type
            print(f"  -> PASS")
            break
        else:
            print(f"  -> FAIL ({(s_prob - s_base)*100:+.1f}pp)")

    if signal_type is None:
        summary["reason"] = "Probabilidad por debajo del threshold"
        if config.get("generate_csv", True):
            generate_oos_csv(spx, models, regime_config, config)
        print_summary(summary); return

    # Webhook
    print(f"\nEnviando webhook: {signal_type}")
    success = send_webhook(signal_type)
    summary["signal_generated"] = True
    summary["reason"] = "Senal enviada" if success else "Senal generada - webhook no configurado"

    if config.get("generate_csv", True):
        generate_oos_csv(spx, models, regime_config, config)

    print_summary(summary)


# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    try:
        generate_signal()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
