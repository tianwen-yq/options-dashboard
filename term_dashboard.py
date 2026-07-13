# -*- coding: utf-8 -*-
"""股指期货期权专业分析仪表盘"""
import sys,os,sqlite3
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# ═══════ Black-Scholes 隐含波动率 ═══════
def bs_price(S,K,T,r,sigma,otype):
    """Black-Scholes 期权定价"""
    if T<=0 or sigma<=0: return 0
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    if otype=="C": return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    else: return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def implied_vol(price,S,K,T,r,otype):
    """二分法求隐含波动率"""
    if price<=0 or T<=0: return np.nan
    lo,hi=0.001,5.0
    for _ in range(50):
        mid=(lo+hi)/2
        p=bs_price(S,K,T,r,mid,otype)
        if abs(p-price)<0.0001: return mid
        if p>price: hi=mid
        else: lo=mid
    return (lo+hi)/2

def bs_delta(S,K,T,r,sigma,otype):
    """Black-Scholes Delta"""
    if T<=0 or sigma<=0: return np.nan
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1) if otype=="C" else norm.cdf(d1)-1

# ═══════ IV 微笑平滑拟合（加权二次多项式）═══════
def smile_fit(strikes, ivs, weights=None):
    """
    加权二次多项式: IV(K) = c0 + c1*K + c2*K²
    平值附近权重大，远端权重小
    """
    if len(strikes) < 4:
        return None
    K = np.array(strikes)
    y = np.array(ivs)
    if weights is None:
        weights = np.ones_like(K)
    K0 = K[len(K)//2]
    dist_w = np.exp(-((K - K0) / (K0 * 0.15)) ** 2)
    w = weights * dist_w
    try:
        coeffs = np.polyfit(K, y, 2, w=w)
        return np.poly1d(coeffs)
    except Exception:
        return None

FUT_NAMES={"IF":"沪深300","IM":"中证1000","IH":"上证50"}
OPT_MAP={"IF":"IO","IH":"HO","IM":"MO"}

# ═══════ 实际波动率计算 ═══════
def calc_realized_vol(close_series, window=20):
    """滚动年化实际波动率，输入收盘价Series，返回Series（小数, 如0.15=15%）"""
    log_ret = np.log(close_series / close_series.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv.round(4)

def calc_vol_cone(prices, windows=[5,10,20,30,60,90,120]):
    """计算波动率锥：各窗口下的历史分位数分布"""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    cone = {}
    for w in windows:
        if len(log_ret) < w:
            continue
        rv = log_ret.rolling(w).std() * np.sqrt(252)
        rv_clean = rv.dropna()
        if len(rv_clean) < 10:
            continue
        cone[w] = {
            "min":  round(float(rv_clean.min()), 4),
            "p10":  round(float(rv_clean.quantile(0.10)), 4),
            "p25":  round(float(rv_clean.quantile(0.25)), 4),
            "p50":  round(float(rv_clean.quantile(0.50)), 4),
            "p75":  round(float(rv_clean.quantile(0.75)), 4),
            "p90":  round(float(rv_clean.quantile(0.90)), 4),
            "max":  round(float(rv_clean.max()), 4),
            "latest": round(float(rv_clean.iloc[-1]), 4),
        }
    return cone

def get_atm_iv_series(opt_code, trading_days, idx_prices):
    """获取近月平值IV时间序列：每个交易日取最近到期月的ATM IV
    idx_prices: dict {date_str: close_price}"""
    conn = sqlite3.connect(cfg.DB_PATH)
    result = []
    r = cfg.RISK_FREE_RATE
    try:
        for td in trading_days:
            spot_p = idx_prices.get(td)
            if spot_p is None:
                continue
            df = pd.read_sql_query(
                "SELECT expiry, strike, otype, close FROM options_daily WHERE code=? AND date=? AND close>0",
                conn, params=(opt_code, td))
            if df.empty:
                continue
            # 近月（距到期≤2天则换次近月）
            exps = sorted(df["expiry"].unique())
            near_exp = exps[0]
            ym0 = str(near_exp)
            T0 = max((datetime(2000+int(ym0[:2]),int(ym0[2:]),15)-datetime.strptime(td,"%Y-%m-%d")).days,1)/365
            if T0 * 365 <= 2 and len(exps) > 1:
                near_exp = exps[1]
            edf = df[df["expiry"] == near_exp].copy()
            if edf.empty:
                continue
            # 找ATM & 算合成远期
            ym = str(near_exp)
            T = max((datetime(2000+int(ym[:2]),int(ym[2:]),15)-datetime.strptime(td,"%Y-%m-%d")).days,1)/365
            edf["dist"] = (edf["strike"] - spot_p).abs()
            atm_rows = edf.nsmallest(2, "dist")
            pm = {}
            for _, r2 in edf.iterrows():
                pm[(r2["strike"], r2["otype"])] = r2["close"]
            fwd_map = {}
            for k in edf["strike"].unique():
                c = pm.get((k, "C")); p = pm.get((k, "P"))
                if c is not None and p is not None:
                    fwd_map[k] = k + (c - p) * np.exp(r * T)
            ivs = []
            for _, row in atm_rows.iterrows():
                F = fwd_map.get(row["strike"], spot_p)
                iv_v = implied_vol(row["close"], F, row["strike"], T, r, row["otype"])
                if not np.isnan(iv_v):
                    ivs.append(iv_v * 100)
            if ivs:
                result.append({"date": td, "atm_iv": round(np.mean(ivs), 2), "expiry": near_exp})
    finally:
        conn.close()
    return result

# ═══════ iVIX 计算（CBOE VIX 方法论） ═══════
def calc_ivix_single(opt_code, date_str):
    """计算单个品种单日的 iVIX 值
    返回: float (iVIX 值) 或 None"""
    conn = sqlite3.connect(cfg.DB_PATH)
    r = cfg.RISK_FREE_RATE
    try:
        # 1. 加载该日该品种所有期权
        df = pd.read_sql_query(
            "SELECT expiry, strike, otype, close FROM options_daily WHERE code=? AND date=? AND close>0 ORDER BY expiry, strike",
            conn, params=(opt_code, date_str))
        if df.empty:
            return None

        # 确定近月和次近月
        exps = sorted(df["expiry"].unique())
        if len(exps) < 2:
            return None

        # 计算每个到期月的剩余自然日
        def nat_days(exp_str, ref_date):
            y, m = 2000 + int(str(exp_str)[:2]), int(str(exp_str)[2:])
            return max((datetime(y, m, 15) - datetime.strptime(ref_date, "%Y-%m-%d")).days, 1)

        exp_days = {e: nat_days(e, date_str) for e in exps}

        # 滚动规则：近月 ≤2 自然日则切换
        near_exp = exps[0]
        if exp_days[near_exp] <= 2:
            near_exp = exps[1]
        next_exp = exps[exps.index(near_exp) + 1] if exps.index(near_exp) + 1 < len(exps) else None
        if next_exp is None:
            return None

        # 2. 分别计算两个月份
        variances = {}
        for exp in [near_exp, next_exp]:
            edf = df[df["expiry"] == exp]
            T = exp_days[exp] / 365.0

            # 获取 C/P 配对
            strikes = sorted(edf["strike"].unique())
            # 构建 price dict: {(strike, otype): close}
            price_map = {}
            for _, row in edf.iterrows():
                price_map[(row["strike"], row["otype"])] = row["close"]

            # 3. 找远期价格 F：最小化 |C-P| 的执行价
            best_diff = float("inf")
            F, K_sel = None, None
            for k in strikes:
                c = price_map.get((k, "C"))
                p = price_map.get((k, "P"))
                if c is None or p is None:
                    continue
                diff = abs(c - p)
                if diff < best_diff:
                    best_diff = diff
                    K_sel = k
                    F = k + np.exp(r * T) * (c - p)

            if F is None:
                continue

            # 4. 确定 K0：低于 F 且最接近 F 的执行价
            k_below = [k for k in strikes if k < F]
            if not k_below:
                k_below = [min(strikes)]
            K0 = max(k_below)

            # 5. 筛选虚值期权，计算 Q(K_i)
            # 截断规则：从K0向两侧，连续两个零报价则截断该侧远端
            zero_count = 0
            put_cutoff = False  # True=看跌侧已被截断
            valid_strikes = []
            for k in strikes:
                if k < K0:
                    if put_cutoff:
                        continue
                    q = price_map.get((k, "P"))
                elif k > K0:
                    q = price_map.get((k, "C"))
                else:
                    cp = price_map.get((k, "C"))
                    pp = price_map.get((k, "P"))
                    q = (cp + pp) / 2 if (cp is not None and pp is not None) else (cp or pp)
                if q is None or q <= 0:
                    zero_count += 1
                    if zero_count >= 2:
                        if k < K0:
                            put_cutoff = True  # 截断看跌侧远端
                        else:
                            break  # 截断看涨侧远端，停止遍历
                else:
                    zero_count = 0
                    valid_strikes.append((k, q))

            if len(valid_strikes) < 3:
                continue

            # 6. 方差求和
            valid_strikes.sort(key=lambda x: x[0])
            variance_sum = 0.0
            for i, (ki, qi) in enumerate(valid_strikes):
                # ΔK
                if i == 0:
                    dk = valid_strikes[1][0] - ki
                elif i == len(valid_strikes) - 1:
                    dk = ki - valid_strikes[i - 1][0]
                else:
                    dk = (valid_strikes[i + 1][0] - valid_strikes[i - 1][0]) / 2.0
                variance_sum += (dk / (ki * ki)) * np.exp(r * T) * qi

            sigma_sq = (2.0 / T) * variance_sum - (1.0 / T) * (F / K0 - 1) ** 2
            if sigma_sq > 0:
                variances[exp] = {"sigma_sq": sigma_sq, "T": T}

        if len(variances) != 2:
            return None

        # 7. 时间插值到 30 自然日
        T1, T2 = variances[near_exp]["T"], variances[next_exp]["T"]
        V1 = T1 * variances[near_exp]["sigma_sq"]
        V2 = T2 * variances[next_exp]["sigma_sq"]
        T_target = 30.0 / 365.0

        if T2 <= T1:
            return None
        V30 = ((T2 - T_target) * V1 + (T_target - T1) * V2) / (T2 - T1)
        sigma_sq_30 = V30 / T_target
        if sigma_sq_30 <= 0:
            return None
        return round(100.0 * np.sqrt(sigma_sq_30), 2)
    finally:
        conn.close()

st.set_page_config(page_title="期权分析",layout="wide")

st.markdown("""<style>
  .stApp{background:#f8f9fa}
  .metric{text-align:center;background:#fafbfc;border:1px solid #eee;border-radius:6px;padding:6px 4px}
  .metric .val{font-size:26px;font-weight:700;color:#1a1a2e;line-height:1.1}
  .metric .chg{font-size:18px;margin-top:2px;line-height:1.1}
  .metric .lbl{font-size:16px;color:#999;line-height:1.1}
  .up{color:#e74c3c}.dn{color:#27ae60}
  .section-title{font-size:16px;font-weight:700;color:#1a1a2e;margin:16px 0 6px 0;padding-bottom:4px;border-bottom:2px solid #e8e8e8}
  div[data-testid="stHorizontalBlock"] button{border-radius:6px!important;font-weight:600!important;font-size:13px!important}
</style>""",unsafe_allow_html=True)

st.title("期权专业分析")

# ═══════ 数据层 ═══════
@st.cache_data(ttl=30)
def Q(sql,params=()):
    conn=sqlite3.connect(cfg.DB_PATH);conn.row_factory=sqlite3.Row
    rows=conn.execute(sql,params).fetchall();conn.close()
    return [dict(r) for r in rows]

@st.cache_data(ttl=30)
def QDF(sql,params=()):
    for _ in range(3):
        try:
            conn=sqlite3.connect(cfg.DB_PATH)
            df=pd.read_sql_query(sql,conn,params=params)
            conn.close()
            return df
        except Exception:
            import time; time.sleep(2)
    return pd.DataFrame()

@st.cache_data(ttl=30)
def LDate(table,code=None):
    if code:
        r=Q(f"SELECT MAX(date) as d FROM {table} WHERE code LIKE ?",(code,)) if "%" in code else Q(f"SELECT MAX(date) as d FROM {table} WHERE code=?",(code,))
    else:
        r=Q(f"SELECT MAX(date) as d FROM {table}")
    return r[0]["d"] if r and r[0]["d"] else None

# ═══════ 品种选择 (期货+期权联动) ═══════
if "u" not in st.session_state: st.session_state.u="IF"
def _switch(v): st.session_state.u=v

all_labels=[("IF沪深300","IF"),("IH上证50","IH"),("IM中证1000","IM"),("iVIX恐慌","iVIX")]
cols=st.columns([1]*len(all_labels))
for col,(label,code) in zip(cols,all_labels):
    with col:
        st.button(label,width='stretch',type="primary" if st.session_state.u==code else "secondary",
                  on_click=_switch,args=(code,))

u=st.session_state.u

# ═══════════════════════════ iVIX 独立板块 ═══════════════════════════
if u == "iVIX":
    st.markdown('<div class="section-title">📊 iVIX 恐慌指数 — 近90天</div>',unsafe_allow_html=True)

    iVIX_CONFIGS = {
        "iVIX-300": {"code": "IO", "color": "#e74c3c", "fill": "rgba(231,76,60,0.08)", "desc": "沪深300"},
        "iVIX-50":  {"code": "HO", "color": "#2980b9", "fill": "rgba(41,128,185,0.08)", "desc": "上证50"},
        "iVIX-1000":{"code": "MO", "color": "#27ae60", "fill": "rgba(39,174,96,0.08)", "desc": "中证1000"},
    }

    @st.cache_data(ttl=300)
    def load_ivix_90d(opt_code):
        conn = sqlite3.connect(cfg.DB_PATH)
        dates = pd.read_sql_query(
            "SELECT DISTINCT date FROM options_daily WHERE code=? ORDER BY date DESC LIMIT 90",
            conn, params=(opt_code,))
        conn.close()
        result = []
        for _, row in dates.iterrows():
            d = str(row["date"])[:10]
            v = calc_ivix_single(opt_code, d)
            if v is not None:
                result.append({"date": d, "ivix": v})
        return sorted(result, key=lambda x: x["date"])

    ivix_cols = st.columns(3)
    for i, (name, icfg) in enumerate(iVIX_CONFIGS.items()):
        with ivix_cols[i]:
            st.markdown(f'<b>{name} — {icfg["desc"]}</b>', unsafe_allow_html=True)
            hist = load_ivix_90d(icfg["code"])
            if hist:
                dates = pd.to_datetime([h["date"] for h in hist])
                vals = [h["ivix"] for h in hist]
                cur = vals[-1] if vals else None
                prv = vals[-2] if len(vals) > 1 else None
                chg_str = f" {cur - prv:+.2f}" if (cur is not None and prv is not None) else ""
                chg_clr = "#e74c3c" if (chg_str and chg_str.startswith(" +")) else "#27ae60" if chg_str else "#999"
                st.markdown(f'<span style="font-size:24px;font-weight:700;color:{icfg["color"]}">{cur}</span>'
                            f'<span style="font-size:14px;color:{chg_clr};margin-left:8px">{chg_str}</span>',
                            unsafe_allow_html=True)

                fig_i = go.Figure()
                # 分位数虚线
                pcts = {"90%": 0.90, "75%": 0.75, "50%": 0.50, "25%": 0.25, "10%": 0.10}
                pct_colors = {"90%": "#e74c3c", "75%": "#e67e22", "50%": "#888",
                              "25%": "#27ae60", "10%": "#2980b9"}
                for label, p in pcts.items():
                    pv = round(np.percentile(vals, p * 100), 2)
                    fig_i.add_hline(y=pv, line_dash="dash", line_color=pct_colors[label],
                        line_width=1, opacity=0.6,
                        annotation_text=f"{label}={pv}", annotation_position="right",
                        annotation_font=dict(size=9, color=pct_colors[label]))
                # 主线
                fig_i.add_trace(go.Scatter(
                    x=dates, y=vals, mode="lines+markers",
                    name=name, line=dict(color=icfg["color"], width=2.5),
                    marker=dict(size=6, color=icfg["color"], symbol="circle",
                                line=dict(width=1, color="#fff"))))
                # 填充区域
                fig_i.add_trace(go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=[0]*len(vals) + vals[::-1],
                    fill="toself", fillcolor=icfg["fill"],
                    mode="none", showlegend=False, hoverinfo="skip"))

                rng = max(vals) - min(vals)
                pad = max(rng * 0.08, 0.5)
                v_lo_i = max(0, min(vals) - pad); v_hi_i = max(vals) + pad
                fig_i.update_layout(
                    template="plotly_white", height=320,
                    margin=dict(l=30, r=10, t=10, b=40),
                    xaxis=dict(title="", showgrid=True, gridcolor="#f0f0f0", dtick=86400000*7, tickformat="%m-%d"),
                    yaxis=dict(title="", showgrid=True, gridcolor="#f0f0f0", range=[v_lo_i, v_hi_i]),
                    paper_bgcolor="#fff", plot_bgcolor="#fff", hovermode="x unified",
                    showlegend=False)
                st.plotly_chart(fig_i, width='stretch', config={"displayModeBar": False})
            else:
                st.info("数据不足")

    # ── 指数价格 & 比值图 ──
    @st.cache_data(ttl=300)
    def load_index_prices():
        conn = sqlite3.connect(cfg.DB_PATH)
        try:
            df300 = pd.read_sql_query("SELECT date,close FROM index_daily WHERE code='000300.SH' ORDER BY date", conn)
            df50  = pd.read_sql_query("SELECT date,close FROM index_daily WHERE code='000016.SH' ORDER BY date", conn)
            df1000= pd.read_sql_query("SELECT date,close FROM index_daily WHERE code='000852.SH' ORDER BY date", conn)
        finally:
            conn.close()
        return df300, df50, df1000

    idx300, idx50, idx1000 = load_index_prices()
    idx300_90 = idx300.tail(90); idx50_90 = idx50.tail(90); idx1000_90 = idx1000.tail(90)
    idx50_180 = idx50.tail(180); idx1000_180 = idx1000.tail(180)

    col_idx, col_ratio = st.columns(2)

    with col_idx:
        st.markdown('<b>指数价格 — 近90天</b>', unsafe_allow_html=True)
        fig_idx = go.Figure()
        fig_idx.add_trace(go.Scatter(x=pd.to_datetime(idx50_90["date"]), y=idx50_90["close"],
            mode="lines", name="上证50", line=dict(color="#1a5276", width=2.5), yaxis="y"))
        fig_idx.add_trace(go.Scatter(x=pd.to_datetime(idx1000_90["date"]), y=idx1000_90["close"],
            mode="lines", name="中证1000", line=dict(color="#e67e22", width=2.5), yaxis="y2"))
        y1 = idx50_90["close"].values; r1 = max(y1)-min(y1); p1 = max(r1*0.1, 1)
        y2 = idx1000_90["close"].values; r2 = max(y2)-min(y2); p2 = max(r2*0.1, 1)
        fig_idx.update_layout(template="plotly_white", height=320,
            margin=dict(l=50,r=50,t=10,b=40),
            legend=dict(orientation="h",y=1.08,font=dict(size=9)),
            xaxis=dict(title="",showgrid=True,gridcolor="#f0f0f0",dtick=86400000*14,tickformat="%m-%d"),
            yaxis=dict(title="上证50",showgrid=False,range=[min(y1)-p1,max(y1)+p1]),
            yaxis2=dict(title="中证1000",showgrid=False,overlaying="y",side="right",
                        range=[min(y2)-p2,max(y2)+p2]),
            paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified")
        st.plotly_chart(fig_idx, width='stretch', config={"displayModeBar": False})

    with col_ratio:
        st.markdown('<b>上证50/中证1000 比值 — 近半年</b>', unsafe_allow_html=True)
        common_dates = set(idx50_180["date"]) & set(idx1000_180["date"])
        ratio_data = []
        for d in sorted(common_dates):
            v50 = float(idx50_180[idx50_180["date"]==d]["close"].iloc[0])
            v1000 = float(idx1000_180[idx1000_180["date"]==d]["close"].iloc[0])
            ratio_data.append({"date": d, "ratio": round(v50/v1000, 6)})
        if ratio_data:
            r_dates = pd.to_datetime([r["date"] for r in ratio_data])
            r_vals = [r["ratio"] for r in ratio_data]
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(x=r_dates, y=r_vals, mode="lines+markers",
                name="50/1000", line=dict(color="#7b2ff7", width=2.5),
                marker=dict(size=4, color="#7b2ff7")))
            for label, p in [("90%",90),("75%",75),("50%",50),("25%",25),("10%",10)]:
                pv = round(np.percentile(r_vals, p), 4)
                fig_ratio.add_hline(y=pv, line_dash="dash", line_color="#999", line_width=1,
                    annotation_text=f"{label}={pv}", annotation_position="right",
                    annotation_font=dict(size=9, color="#888"))
            rng = max(r_vals)-min(r_vals); pad = max(rng*0.08, 0.001)
            fig_ratio.update_layout(template="plotly_white", height=320,
                margin=dict(l=30,r=20,t=10,b=40),
                xaxis=dict(title="",showgrid=True,gridcolor="#f0f0f0",dtick=86400000*30,tickformat="%m-%d"),
                yaxis=dict(title="比值",showgrid=True,gridcolor="#f0f0f0",
                           range=[min(r_vals)-pad,max(r_vals)+pad]),
                paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified")
            st.plotly_chart(fig_ratio, width='stretch', config={"displayModeBar": False})
        else:
            st.info("数据不足")

    st.stop()

opt=OPT_MAP[u]
idx_code=cfg.FUTURES_TO_INDEX[u]

LD=LDate("futures_daily",f"{u}%"); OLD=LDate("options_daily",opt)
if not LD or not OLD: st.warning("请先运行数据管理获取数据");st.stop()

# ── 加载期货数据 ──
FT=QDF("SELECT * FROM futures_daily WHERE code LIKE ? AND date=? ORDER BY code",(f"{u}%",LD))
FY=QDF("SELECT * FROM futures_daily WHERE code LIKE ? AND date=(SELECT MAX(date) FROM futures_daily WHERE code LIKE ? AND date<?)",(f"{u}%",f"{u}%",LD))

# 标的价格：指数收盘价
IC=Q("SELECT close FROM index_daily WHERE code=? AND date=?",(idx_code,LD))
if not IC: st.error(f"数据库无指数数据"); st.stop()
spot=float(IC[0]["close"])
PD_LD=Q("SELECT MAX(date) as d FROM index_daily WHERE code=? AND date<?",(idx_code,LD))
if not PD_LD or not PD_LD[0]["d"]: st.error(f"无前一日指数"); st.stop()
IDX_PREV_ROW=Q("SELECT close FROM index_daily WHERE code=? AND date=?",(idx_code,PD_LD[0]["d"]))
IDX_PREV=float(IDX_PREV_ROW[0]["close"]) if IDX_PREV_ROW else spot

@st.cache_data(ttl=30)
def load_options(opt_code,date_str,spot_p,prev_spot_p):
    """加载期权+合成期货IV"""
    conn=sqlite3.connect(cfg.DB_PATH)
    df=pd.read_sql_query("SELECT * FROM options_daily WHERE code=? AND date=?",conn,params=(opt_code,date_str))
    conn.close()
    if df.empty: return df,None
    r = cfg.RISK_FREE_RATE
    # 按(到期月,行权价)配对Call/Put，算合成期货F_hat
    ivs = [np.nan]*len(df)
    fwd_map = {}  # (expiry, strike) -> F_hat
    for exp in df["expiry"].unique():
        edf = df[df["expiry"]==exp]
        ym = str(exp)
        T = max((datetime(2000+int(ym[:2]),int(ym[2:]),15)-datetime.strptime(date_str,"%Y-%m-%d")).days,1)/365
        for k in edf["strike"].unique():
            c_rows = edf[(edf["strike"]==k)&(edf["otype"]=="C")]
            p_rows = edf[(edf["strike"]==k)&(edf["otype"]=="P")]
            if len(c_rows)>0 and len(p_rows)>0 and c_rows["close"].values[0]>0 and p_rows["close"].values[0]>0:
                cp = c_rows["close"].values[0]
                pp = p_rows["close"].values[0]
                F_hat = k + (cp - pp) * np.exp(r*T)
                fwd_map[(exp,k)] = F_hat
    # 对每个期权计算IV
    # 先按到期月算平值范围（均值F_hat ±30%），范围外不计算IV
    atm_range = {}
    for exp in df["expiry"].unique():
        fwds = [fv for (e,sk),fv in fwd_map.items() if e==exp]
        if fwds:
            atm_f = sum(fwds)/len(fwds)
            atm_range[exp] = (atm_f*0.7, atm_f*1.3)
    fwds_col = [np.nan]*len(df)
    for i,(_,r_row) in enumerate(df.iterrows()):
        if r_row["close"]<=0 or r_row["strike"]<=0: continue
        exp = r_row["expiry"]; k = r_row["strike"]
        lo, hi = atm_range.get(exp, (0, 1e9))
        if k < lo or k > hi: continue
        ym = str(exp)
        T = max((datetime(2000+int(ym[:2]),int(ym[2:]),15)-datetime.strptime(date_str,"%Y-%m-%d")).days,1)/365
        F = fwd_map.get((exp,k))
        if F is None:
            near_fwds = [fv for (e,sk),fv in fwd_map.items() if e==exp]
            F = sum(near_fwds)/len(near_fwds) if near_fwds else spot_p
        fwds_col[i] = F
        ivs[i] = implied_vol(r_row["close"],F,k,T,r,r_row["otype"])
    df["iv"] = [v*100 if v==v else np.nan for v in ivs]
    df["fwd"] = fwds_col
    # 前一日
    conn2=sqlite3.connect(cfg.DB_PATH)
    try:
        pr=conn2.execute("SELECT MAX(date) FROM options_daily WHERE code=? AND date<?",(opt_code,date_str)).fetchone()
        if pr and pr[0]:
            pdf=pd.read_sql_query("SELECT * FROM options_daily WHERE code=? AND date=?",conn2,params=(opt_code,pr[0]))
            ivs2 = [np.nan]*len(pdf)
            fwd_map2 = {}
            for exp in pdf["expiry"].unique():
                edf = pdf[pdf["expiry"]==exp]
                ym = str(exp)
                T = max((datetime(2000+int(ym[:2]),int(ym[2:]),15)-datetime.strptime(pr[0],"%Y-%m-%d")).days,1)/365
                for k in edf["strike"].unique():
                    c_rows = edf[(edf["strike"]==k)&(edf["otype"]=="C")]
                    p_rows = edf[(edf["strike"]==k)&(edf["otype"]=="P")]
                    if len(c_rows)>0 and len(p_rows)>0 and c_rows["close"].values[0]>0 and p_rows["close"].values[0]>0:
                        cp = c_rows["close"].values[0]; pp = p_rows["close"].values[0]
                        fwd_map2[(exp,k)] = k + (cp - pp)*np.exp(r*T)
            atm_range2 = {}
            for exp in pdf["expiry"].unique():
                fwds = [fv for (e,sk),fv in fwd_map2.items() if e==exp]
                if fwds:
                    atm_f = sum(fwds)/len(fwds)
                    atm_range2[exp] = (atm_f*0.7, atm_f*1.3)
            fwds_col2 = [np.nan]*len(pdf)
            for i,(_,r_row) in enumerate(pdf.iterrows()):
                if r_row["close"]<=0 or r_row["strike"]<=0: continue
                exp = r_row["expiry"]; k = r_row["strike"]
                lo, hi = atm_range2.get(exp, (0, 1e9))
                if k < lo or k > hi: continue
                ym = str(exp)
                T = max((datetime(2000+int(ym[:2]),int(ym[2:]),15)-datetime.strptime(pr[0],"%Y-%m-%d")).days,1)/365
                F = fwd_map2.get((exp,k))
                if F is None:
                    near_fwds = [fv for (e,sk),fv in fwd_map2.items() if e==exp]
                    F = sum(near_fwds)/len(near_fwds) if near_fwds else prev_spot_p
                fwds_col2[i] = F
                ivs2[i] = implied_vol(r_row["close"],F,k,T,r,r_row["otype"])
            pdf["iv"] = [v*100 if v==v else np.nan for v in ivs2]
            pdf["fwd"] = fwds_col2
    finally:
        conn2.close()
    return df,pdf if pr and pr[0] else None

OD,PD=load_options(opt,OLD,spot,IDX_PREV)

# ═══════════════════════════ 0. 市场概览 ═══════════════════════════
st.markdown('<div class="section-title">📌 市场概览</div>',unsafe_allow_html=True)
calls=OD[OD["otype"]=="C"]; puts=OD[OD["otype"]=="P"]
tco,tpo=int(calls["open_interest"].sum()),int(puts["open_interest"].sum())
tcv,tpv=int(calls["volume"].sum()),int(puts["volume"].sum())
pcr_vol=round(tpv/tcv,2) if tcv>0 else 0; pcr_oi=round(tpo/tco,2) if tco>0 else 0
# Call/Put OI 对比前日
prev_call_oi = int(PD[PD["otype"]=="C"]["open_interest"].sum()) if PD is not None and not PD.empty else 0
prev_put_oi  = int(PD[PD["otype"]=="P"]["open_interest"].sum()) if PD is not None and not PD.empty else 0
call_arrow = ' <span style="color:#e74c3c">▲</span>' if tco>prev_call_oi else (' <span style="color:#27ae60">▼</span>' if tco<prev_call_oi else '')
put_arrow  = ' <span style="color:#e74c3c">▲</span>' if tpo>prev_put_oi  else (' <span style="color:#27ae60">▼</span>' if tpo<prev_put_oi  else '')
# 最大痛点: 使所有期权内在价值之和最小的行权价(仅算近月)
max_pain_strike = None
max_pain_oi = None
near_exp_mp = sorted(OD["expiry"].unique())[0] if len(OD["expiry"].unique())>0 else None
if near_exp_mp:
    ndf = OD[OD["expiry"]==near_exp_mp]
    strikes = sorted(ndf["strike"].unique())
    min_pain = float('inf')
    for K in strikes:
        pain = 0
        for _, r in ndf.iterrows():
            oi = r["open_interest"]
            if r["otype"] == "C":
                pain += oi * max(0, K - r["strike"])
            else:
                pain += oi * max(0, r["strike"] - K)
        if pain < min_pain:
            min_pain = pain
            max_pain_strike = int(K)
            max_pain_oi = int(ndf[ndf["strike"]==K]["open_interest"].sum())
# 近月基差
first_fut=FT.iloc[0] if len(FT)>0 else None
basis_v=round(first_fut["close"]-spot,1) if first_fut is not None else None
basis_pct=round(basis_v/spot*100,2) if basis_v is not None and spot>0 else None
# 偏度信号
near_exp=sorted(OD["expiry"].unique())[0] if len(OD["expiry"].unique())>0 else None
skew_signal="—"
if near_exp:
    ne=OD[OD["expiry"]==near_exp]; nec=ne[ne["otype"]=="C"]; nep=ne[ne["otype"]=="P"]
    if not nec.empty and not nep.empty:
        atm_i=(nec["strike"]-spot).abs().idxmin()
        atm_iv_2=nec.loc[atm_i,"iv"] if atm_i in nec.index else np.nan
        otm_p=nep[(nep["strike"]<spot*0.98)&(nep["strike"]>spot*0.85)]
        otm_iv_2=otm_p["iv"].mean() if len(otm_p)>0 else atm_iv_2
        skew_signal="Put偏贵⚠️" if otm_iv_2>atm_iv_2 else "正常"

cols=st.columns(7)
metrics=[
    ("标的",f"{spot:.1f}",""),
    ("基差",f"{basis_v:.1f}" if basis_v is not None else "N/A",f"{basis_pct:+.2f}%" if basis_pct is not None else ""),
    ("PCR(Vol)",f"{pcr_vol:.2f}",""),
    ("PCR(OI)",f"{pcr_oi:.2f}",""),
    ("Call OI",f"{tco/1e4:.1f}万{call_arrow}",""),
    ("Put OI",f"{tpo/1e4:.1f}万{put_arrow}",""),
    ("最大痛点",f"{max_pain_strike}" if max_pain_strike is not None else "N/A",f"OI:{max_pain_oi/1e4:.1f}万" if max_pain_oi is not None else ""),
]
for i,(l,v,s) in enumerate(metrics):
    with cols[i]:
        st.markdown(f'<div class="metric"><div class="lbl">{l}</div><div class="val">{v}</div><div class="chg">{s}</div></div>',unsafe_allow_html=True)

# ═══════════════════════════ 1. 期货期限结构 ═══════════════════════════
st.markdown('<div class="section-title">📈 期货期限结构</div>',unsafe_allow_html=True)
months=[]; ym={}
if not FY.empty:
    for _,r in FY.iterrows(): ym[r["code"]]=r["close"]
total_oi_all=0; total_pre_oi=0
for _,r in FT.iterrows():
    code=r["code"]; m=code[2:]
    pc=ym.get(code,r["close"])
    chg=(r["close"]-pc)/pc*100 if pc else 0
    oi=int(r["open_interest"]); pre_oi=int(r["pre_open_interest"])
    total_oi_all+=oi; total_pre_oi+=pre_oi
    months.append({"code":code,"month":m,"close":r["close"],"oi":oi,"chg":chg,"pre_oi":pre_oi,"basis":r["basis"],"basis_rate":r["basis_rate"]})

cards=st.columns(len(months)+1)
with cards[0]:
    st.markdown(f'<div class="metric"><div class="lbl">标的</div><div class="val">{spot:.1f}</div></div>',unsafe_allow_html=True)
for i,m in enumerate(months):
    cls="dn" if m["chg"]<0 else "up"
    with cards[i+1]:
        st.markdown(f'<div class="metric"><div class="lbl">{m["month"]}</div><div class="val">{m["close"]:.1f}</div><div class="chg {cls}">{m["chg"]:+.2f}%</div></div>',unsafe_allow_html=True)

cats=["指数"]+[m["month"] for m in months]
prices=[spot]+[m["close"] for m in months]
prices_prev=[IDX_PREV]+[ym.get(m["code"]) for m in months]
fut_oi=[0]+[m["oi"] for m in months]
fut_oi_delta=[0]+[m["oi"]-m["pre_oi"] for m in months]
basis_vals=[None]+[m["basis"] for m in months]

n_cats = len(cats)
x_cats = list(range(n_cats))
col_t1,col_t2=st.columns(2)
with col_t1:
    st.markdown('<b>价格 & OI变化</b><br><small>浅色柱=当日OI 深色柱=增减量(水上增/水下减)</small>',unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_cats,y=prices,mode="lines+markers",name="最新",
        line=dict(color="#5b6abf",width=2.5),marker=dict(size=6),yaxis="y"))
    fig1.add_trace(go.Scatter(x=x_cats,y=prices_prev,mode="lines+markers",name="上日",
        line=dict(color="#8899aa",width=1.8,dash="dot"),marker=dict(size=5,color="#8899aa"),yaxis="y"))
    fig1.add_trace(go.Bar(x=x_cats,y=fut_oi,name="合约OI",
        marker_color="#d5d5d5",opacity=0.5,width=0.36,yaxis="y2"))
    fig1.add_trace(go.Bar(x=x_cats,y=fut_oi_delta,name="OI增减",
        marker_color=["#ddd"]+["#e74c3c" if v>0 else "#27ae60" if v<0 else "#ddd" for v in fut_oi_delta[1:]],
        opacity=0.85,width=0.36,yaxis="y2"))
    fig1.update_layout(template="plotly_white",height=300,margin=dict(l=40,r=60,t=30,b=40),
        legend=dict(orientation="h",y=1.05,font=dict(size=9)),
        xaxis=dict(showgrid=False,tickvals=x_cats,ticktext=cats),
        yaxis=dict(title="价格",showgrid=False),
        yaxis2=dict(title="持仓量",showgrid=False,overlaying="y",side="right"),
        paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified",
        barmode="overlay")
    st.plotly_chart(fig1,width='stretch',config={"displayModeBar":False})

with col_t2:
    near_code = months[0]["code"] if months else None
    if near_code:
        st.markdown(f'<b>近月基差 & 持仓量</b><br><small>{u} 近60天(自动换月)</small>',unsafe_allow_html=True)
        import sqlite3 as _sql
        _conn=_sql.connect(cfg.DB_PATH)
        basis_hist=pd.read_sql_query("SELECT f.date,f.code,f.basis,f.open_interest FROM futures_daily f INNER JOIN (SELECT date,MIN(code) as mc FROM futures_daily WHERE code LIKE ? GROUP BY date) g ON f.date=g.date AND f.code=g.mc WHERE f.date>=(SELECT MIN(date) FROM (SELECT DISTINCT date FROM futures_daily WHERE code LIKE ? ORDER BY date DESC LIMIT 60)) ORDER BY f.date",_conn,params=(f"{u}%",f"{u}%"))
        _conn.close()
        if not basis_hist.empty:
            basis_hist = basis_hist.sort_values("date")
            b_dates = [str(d)[:10] for d in basis_hist["date"]]
            b_vals = list(basis_hist["basis"])
            b_oi = list(basis_hist["open_interest"])
            fig_b = make_subplots(specs=[[{"secondary_y": True}]])
            fig_b.add_trace(go.Scatter(x=b_dates, y=b_vals, mode="lines+markers",
                name="基差(点)", line=dict(color="#e74c3c", width=2), marker=dict(size=5)), secondary_y=False)
            fig_b.add_trace(go.Bar(x=b_dates, y=b_oi, name="持仓量",
                marker_color="#b0c4de", opacity=0.5, width=0.3), secondary_y=True)
            fig_b.add_hline(y=0, line_dash="dash", line_color="#999")
            fig_b.update_layout(template="plotly_white", height=300,
                margin=dict(l=40, r=60, t=30, b=40),
                legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                xaxis=dict(showgrid=False, type="category"),
                yaxis=dict(title="基差(点)", showgrid=False),
                yaxis2=dict(title="持仓量", showgrid=False),
                paper_bgcolor="#fff", plot_bgcolor="#fff", hovermode="x unified")
            st.plotly_chart(fig_b, width='stretch', config={"displayModeBar": False})
        else:
            st.info("无基差历史数据")
    else:
        st.info("无近月合约")

# ═══════════════════════════ 2. PCR趋势 + OI集中度 ═══════════════════════════
st.markdown('<div class="section-title">📊 期权情绪 & 资金流向</div>',unsafe_allow_html=True)
col_l,col_r=st.columns(2)

with col_l:
    # PCR趋势
    st.markdown('<b>PCR趋势 (近90天)</b>',unsafe_allow_html=True)
    pcr_data=QDF("SELECT * FROM options_summary WHERE code=? ORDER BY date DESC LIMIT 90",(opt,))
    if not pcr_data.empty:
        pcr_data=pcr_data.sort_values("date")
        pcr_dates=[str(d)[:10] for d in pcr_data["date"]]
        fig_pcr=make_subplots(specs=[[{"secondary_y":False}]])
        fig_pcr.add_trace(go.Scatter(x=pcr_dates,y=pcr_data["pcr_volume"],mode="lines+markers",name="PCR(Vol)",line=dict(color="#e74c3c",width=2)))
        fig_pcr.add_trace(go.Scatter(x=pcr_dates,y=pcr_data["pcr_oi"],mode="lines+markers",name="PCR(OI)",line=dict(color="#5b6abf",width=2)))
        fig_pcr.add_hline(y=1,line_dash="dash",line_color="#999",annotation_text="1.0")
        fig_pcr.update_layout(template="plotly_white",height=300,margin=dict(l=30,r=10,t=10,b=40),legend=dict(orientation="h",y=1.05,font=dict(size=10)),xaxis=dict(showgrid=False,type="category",tickformat="%Y-%m-%d"),yaxis=dict(title="PCR",showgrid=False),paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified")
        st.plotly_chart(fig_pcr,width='stretch',config={"displayModeBar":False})
    else:
        st.info("需多天数据")

with col_r:
    # OI集中度 (全部行权价)
    st.markdown('<b>OI集中度</b>',unsafe_allow_html=True)
    oi_by_strike=OD.groupby("strike").agg(call_oi=("open_interest",lambda x:x[OD["otype"]=="C"].sum()),put_oi=("open_interest",lambda x:x[OD["otype"]=="P"].sum())).reset_index()
    oi_by_strike=oi_by_strike.sort_values("strike",ascending=True)
    fig_conc=go.Figure()
    fig_conc.add_trace(go.Bar(x=oi_by_strike["strike"].astype(str),y=oi_by_strike["call_oi"],name="Call OI",marker_color="#e74c3c",opacity=0.7))
    fig_conc.add_trace(go.Bar(x=oi_by_strike["strike"].astype(str),y=oi_by_strike["put_oi"],name="Put OI",marker_color="#27ae60",opacity=0.7))
    fig_conc.update_layout(template="plotly_white",height=300,margin=dict(l=30,r=10,t=10,b=40),barmode="stack",legend=dict(orientation="h",y=1.05,font=dict(size=10)),xaxis=dict(title="行权价",showgrid=False,tickangle=-45),yaxis=dict(title="持仓量",showgrid=False),paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified")
    st.plotly_chart(fig_conc,width='stretch',config={"displayModeBar":False})

# ═══════════════════════════ 3. 平值IV + 偏度 ═══════════════════════════
st.markdown('<div class="section-title">📉 定价分析</div>',unsafe_allow_html=True)
col_iv,col_sk=st.columns(2)

with col_iv:
    st.markdown('<b>平值IV期限结构</b><br><small>ATM IV = (Call IV + Put IV) / 2  |  浅色=当日OI 深色=增减量(水上增/水下减)</small>',unsafe_allow_html=True)
    # 专业算法：每到期月取ATM行权价，用Call和Put的IV取平均
    atm_iv_data=[]
    prev_atm_map={}
    for exp in sorted(OD["expiry"].unique()):
        edf=OD[OD["expiry"]==exp]; ec=edf[edf["otype"]=="C"]; ep=edf[edf["otype"]=="P"]
        if ec.empty or ep.empty: continue
        atm_idx = (ec["strike"]-spot).abs().values.argmin()
        atm_strike = ec.iloc[atm_idx]["strike"]
        c_iv = ec.iloc[atm_idx]["iv"]
        p_row = ep[ep["strike"]==atm_strike]
        p_iv = p_row["iv"].iloc[0] if len(p_row)>0 else c_iv
        atm_iv=round((c_iv+p_iv)/2,2) if c_iv==c_iv and p_iv==p_iv else None
        call_oi = int(ec["open_interest"].sum())
        put_oi = int(ep["open_interest"].sum())
        if atm_iv:
            atm_iv_data.append({"expiry":exp,"atm_iv":atm_iv,"strike":atm_strike,
                               "call_oi":call_oi,"put_oi":put_oi,
                               "prev_call_oi":0,"prev_put_oi":0})
        # 前一日
        if PD is not None and not PD.empty:
            pedf=PD[PD["expiry"]==exp]; pec=pedf[pedf["otype"]=="C"]; pep=pedf[pedf["otype"]=="P"]
            if atm_iv:
                atm_iv_data[-1]["prev_call_oi"] = int(pec["open_interest"].sum()) if not pec.empty else 0
                atm_iv_data[-1]["prev_put_oi"] = int(pep["open_interest"].sum()) if not pep.empty else 0
            if not pec.empty and not pep.empty:
                pa = (pec["strike"]-IDX_PREV).abs().values.argmin()
                pas = pec.iloc[pa]["strike"]
                pc_iv = pec.iloc[pa]["iv"]
                pp_row = pep[pep["strike"]==pas]
                pp_iv=pp_row["iv"].iloc[0] if len(pp_row)>0 else pc_iv
                prev_atm=round((pc_iv+pp_iv)/2,2) if pc_iv==pc_iv and pp_iv==pp_iv else None
                if prev_atm: prev_atm_map[exp]=prev_atm

    if atm_iv_data:
        iv_df=pd.DataFrame(atm_iv_data)
        # ATM IV 数据表
        atm_tbl_cols = st.columns(len(iv_df)+1)
        with atm_tbl_cols[0]:
            st.markdown('<div style="font-size:11px;color:#999;text-align:center;padding-top:8px">指标</div>',unsafe_allow_html=True)
        for i, row in iv_df.iterrows():
            with atm_tbl_cols[i+1]:
                st.markdown(f'<div style="text-align:center;font-weight:700;font-size:13px;color:#7b2ff7">{row["expiry"]}</div>',unsafe_allow_html=True)
        with atm_tbl_cols[0]:
            st.markdown('<div style="font-size:11px;color:#666;text-align:center">当前</div>',unsafe_allow_html=True)
        for i, row in iv_df.iterrows():
            with atm_tbl_cols[i+1]:
                st.markdown(f'<div style="text-align:center;font-size:14px;font-weight:600">{row["atm_iv"]:.1f}%</div>',unsafe_allow_html=True)
        with atm_tbl_cols[0]:
            st.markdown('<div style="font-size:11px;color:#666;text-align:center">日差</div>',unsafe_allow_html=True)
        for i, row in iv_df.iterrows():
            prev_v = prev_atm_map.get(row["expiry"])
            diff_v = row["atm_iv"] - prev_v if prev_v is not None else None
            diff_str = f'{diff_v:+.1f}' if diff_v is not None else "—"
            color = "#e74c3c" if (diff_v is not None and diff_v > 0) else "#27ae60" if (diff_v is not None and diff_v < 0) else "#999"
            with atm_tbl_cols[i+1]:
                st.markdown(f'<div style="text-align:center;font-size:12px;color:{color}">{diff_str}</div>',unsafe_allow_html=True)
        # 计算IV纵轴范围
        iv_vals = iv_df["atm_iv"].dropna()
        prev_iv_vals = [prev_atm_map.get(e) for e in iv_df["expiry"] if prev_atm_map.get(e) is not None]
        all_iv = list(iv_vals) + prev_iv_vals
        if all_iv:
            iv_min, iv_max = min(all_iv), max(all_iv)
            iv_pad = max((iv_max - iv_min) * 0.15, 1.0)
            iv_lo = max(0, iv_min - iv_pad)
            iv_hi = iv_max + iv_pad
        else:
            iv_lo, iv_hi = 0, 30

        fig_iv=make_subplots(specs=[[{"secondary_y":True}]])
        # 统一用数字x坐标，手动设tick标签
        n = len(iv_df)
        x_num = list(range(n))
        bar_w = 0.18
        call_x = [i - bar_w for i in x_num]
        put_x  = [i + bar_w for i in x_num]
        exp_labels = list(iv_df["expiry"])
        # IV线 — 也用数字x
        fig_iv.add_trace(go.Scatter(x=x_num, y=iv_df["atm_iv"], mode="lines+markers", name="ATM IV",
            line=dict(color="#7b2ff7", width=2.5), marker=dict(size=6)), secondary_y=False)
        prev_y = [prev_atm_map.get(e, None) for e in exp_labels]
        fig_iv.add_trace(go.Scatter(x=x_num, y=prev_y, mode="lines+markers", name="上日ATM IV",
            line=dict(color="#888", width=1.5, dash="dot"), marker=dict(size=5)), secondary_y=False)
        # Call OI: 浅红=当日总OI + 深红=增减量(从零轴)
        fig_iv.add_trace(go.Bar(x=call_x, y=iv_df["call_oi"], name="Call OI",
            marker_color="#f5b7b1", opacity=0.5, width=bar_w*2), secondary_y=True)
        fig_iv.add_trace(go.Bar(x=call_x, y=iv_df["call_oi"]-iv_df["prev_call_oi"], name="Call增减",
            marker_color="#e74c3c", opacity=0.9, width=bar_w*2), secondary_y=True)
        # Put OI: 浅绿=当日总OI + 深绿=增减量(从零轴)
        fig_iv.add_trace(go.Bar(x=put_x, y=iv_df["put_oi"], name="Put OI",
            marker_color="#a9dfbf", opacity=0.5, width=bar_w*2), secondary_y=True)
        fig_iv.add_trace(go.Bar(x=put_x, y=iv_df["put_oi"]-iv_df["prev_put_oi"], name="Put增减",
            marker_color="#27ae60", opacity=0.9, width=bar_w*2), secondary_y=True)
        fig_iv.update_layout(template="plotly_white", height=320, margin=dict(l=30,r=60,t=30,b=40),
            legend=dict(orientation="h", y=1.08, font=dict(size=9)),
            xaxis=dict(showgrid=False, tickvals=x_num, ticktext=exp_labels),
            yaxis=dict(title="IV%", showgrid=False, range=[iv_lo, iv_hi]),
            yaxis2=dict(title="持仓量", showgrid=False),
            paper_bgcolor="#fff", plot_bgcolor="#fff",
            barmode="overlay", hovermode="x unified")
        st.plotly_chart(fig_iv,width='stretch',config={"displayModeBar":False})
    else:
        st.info("无数据")

with col_sk:
    st.markdown('<b>隐波偏度的期限结构</b><br><small>偏度 = IV(Put Δ=-0.25) − IV(Call Δ=+0.25)</small>',unsafe_allow_html=True)
    
    def find_delta_iv(edf, target_delta, spot_p, date_str):
        """找到最接近target_delta的期权并返回其IV(小数)，用每张期权的合成期货F̂"""
        best_iv, best_diff = np.nan, 1e9
        for _, r in edf.iterrows():
            iv_v = r["iv"]
            if pd.isna(iv_v) or iv_v <= 0: continue
            ym_s = str(r["expiry"])
            T_d = max((datetime(2000+int(ym_s[:2]), int(ym_s[2:]), 15) - 
                       datetime.strptime(date_str, "%Y-%m-%d")).days, 1) / 365
            F_val = r["fwd"] if "fwd" in r and not pd.isna(r["fwd"]) else spot_p
            d = bs_delta(F_val, r["strike"], T_d, cfg.RISK_FREE_RATE, iv_v / 100, r["otype"])
            if abs(d - target_delta) < best_diff:
                best_diff = abs(d - target_delta)
                best_iv = iv_v
        return best_iv / 100 if (best_diff < 0.5 and not pd.isna(best_iv)) else np.nan
    
    def days_to_expiry(exp_str, date_str):
        return max((datetime(2000+int(exp_str[:2]), int(exp_str[2:]), 15) - 
                    datetime.strptime(date_str, "%Y-%m-%d")).days, 1)
    
    skew_data = []
    prev_skew_map = {}
    exp_labels = []
    
    for exp in sorted(OD["expiry"].unique()):
        edf = OD[OD["expiry"] == exp]
        ec = edf[edf["otype"] == "C"]
        ep = edf[edf["otype"] == "P"]
        if ec.empty or ep.empty: continue
        
        call_25 = find_delta_iv(ec, 0.25, spot, OLD)
        put_25 = find_delta_iv(ep, -0.25, spot, OLD)
        sk = round(put_25 - call_25, 4) if (not pd.isna(call_25) and not pd.isna(put_25)) else np.nan
        dte = days_to_expiry(exp, OLD)
        if not pd.isna(sk):
            skew_data.append({"expiry": exp, "skew": sk, "dte": dte})
        exp_labels.append(exp)
        
        # 前一日
        if PD is not None and not PD.empty:
            pedf = PD[PD["expiry"] == exp]
            pec = pedf[pedf["otype"] == "C"]
            pep = pedf[pedf["otype"] == "P"]
            if not pec.empty and not pep.empty:
                prev_date = str(PD.iloc[0]["date"])[:10]
                pc25 = find_delta_iv(pec, 0.25, IDX_PREV, prev_date)
                pp25 = find_delta_iv(pep, -0.25, IDX_PREV, prev_date)
                psk_v = round(pp25 - pc25, 4) if (not pd.isna(pc25) and not pd.isna(pp25)) else None
                if psk_v is not None:
                    prev_skew_map[exp] = psk_v
    
    if skew_data:
        sk_df = pd.DataFrame(skew_data)
        # 数据表
        tbl_cols = st.columns(len(sk_df) + 1)
        with tbl_cols[0]:
            st.markdown('<div style="font-size:11px;color:#999;text-align:center;padding-top:8px">指标</div>', unsafe_allow_html=True)
        for i, row in sk_df.iterrows():
            with tbl_cols[i+1]:
                st.markdown(f'<div style="text-align:center;font-weight:700;font-size:13px;color:#7b2ff7">{row["expiry"]}</div>', unsafe_allow_html=True)
        # 现值行
        with tbl_cols[0]:
            st.markdown('<div style="font-size:11px;color:#666;text-align:center">現值</div>', unsafe_allow_html=True)
        for i, row in sk_df.iterrows():
            with tbl_cols[i+1]:
                st.markdown(f'<div style="text-align:center;font-size:14px;font-weight:600">{row["skew"]:.4f}</div>', unsafe_allow_html=True)
        # 日差行
        with tbl_cols[0]:
            st.markdown('<div style="font-size:11px;color:#666;text-align:center">日差</div>', unsafe_allow_html=True)
        for i, row in sk_df.iterrows():
            prev_v = prev_skew_map.get(row["expiry"])
            diff_str = f'{row["skew"] - prev_v:+.4f}' if prev_v is not None else "—"
            color = "#e74c3c" if (prev_v is not None and row["skew"] > prev_v) else "#27ae60" if (prev_v is not None and row["skew"] < prev_v) else "#999"
            with tbl_cols[i+1]:
                st.markdown(f'<div style="text-align:center;font-size:12px;color:{color}">{diff_str}</div>', unsafe_allow_html=True)
        
        # 期限结构图 (X轴=距到期天数)
        fig_sk = go.Figure()
        fig_sk.add_trace(go.Scatter(
            x=sk_df["dte"], y=sk_df["skew"],
            mode="lines+markers", name="今日偏度",
            line=dict(color="#7b2ff7", width=2.5),
            marker=dict(size=8, color="#7b2ff7"),
            text=sk_df["expiry"], textposition="top center",
            textfont=dict(size=10, color="#7b2ff7")
        ))
        prev_dtes = [days_to_expiry(e, OLD) for e in sk_df["expiry"]]
        prev_vals = [prev_skew_map.get(e) for e in sk_df["expiry"]]
        fig_sk.add_trace(go.Scatter(
            x=prev_dtes, y=prev_vals,
            mode="lines+markers", name="上日偏度",
            line=dict(color="#b0a0d0", width=1.8, dash="dot"),
            marker=dict(size=6, color="#b0a0d0")
        ))
        fig_sk.add_hline(y=0, line_dash="dash", line_color="#ddd", annotation_text="零偏度线")
        # 计算偏度纵轴范围，紧凑不留白
        sk_vals = list(sk_df["skew"]) + [v for v in prev_vals if v is not None]
        if sk_vals:
            sk_min, sk_max = min(sk_vals), max(sk_vals)
            sk_pad = max((sk_max - sk_min) * 0.15, 0.02)
            sk_lo = min(0, sk_min) - sk_pad
            sk_hi = max(0, sk_max) + sk_pad
        else:
            sk_lo, sk_hi = -0.15, 0.15
        fig_sk.update_layout(
            template="plotly_white", height=320,
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.08, font=dict(size=9)),
            xaxis=dict(title="距到期天数", showgrid=False, dtick=10),
            yaxis=dict(title="偏度", showgrid=False, tickformat=".4f", range=[sk_lo, sk_hi]),
            paper_bgcolor="#fff", plot_bgcolor="#fff", hovermode="x unified"
        )
        st.plotly_chart(fig_sk, width='stretch', config={"displayModeBar": False})
    else:
        st.info("数据不足，无法计算偏度")

# ═══════════════════════════ 3.5 波动率分析 ═══════════════════════════
st.markdown('<div class="section-title">📊 波动率分析</div>',unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_vol_data(idx_code, opt_code):
    """加载波动率分析所需数据"""
    conn = sqlite3.connect(cfg.DB_PATH)
    try:
        # 指数日线（全量，用于波动率锥）
        idx_all = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE code=? ORDER BY date",
            conn, params=(idx_code,))
        idx_all["date"] = pd.to_datetime(idx_all["date"])
        idx_all = idx_all.set_index("date")["close"]
        # 最近100个交易日用于RV
        idx_recent = idx_all.tail(100)
        # 计算20天RV
        rv = calc_realized_vol(idx_recent, 20)
        rv = rv.tail(60)  # 显示最近60天
        # 波动率锥
        vol_cone = calc_vol_cone(idx_all)
        # 当前最新RV (各窗口)
        current_rv = {w: v["latest"] for w, v in vol_cone.items()} if vol_cone else {}
        # 近月ATM IV时间序列（最近60个交易日）
        trading_days = sorted(idx_recent.index[-60:].strftime("%Y-%m-%d"))
        idx_price_map = {d.strftime("%Y-%m-%d"): float(v) for d, v in idx_recent.items()}
        atm_iv_list = get_atm_iv_series(opt_code, trading_days, idx_price_map)
    finally:
        conn.close()
    return rv, vol_cone, current_rv, atm_iv_list, idx_recent

rv_series, vol_cone, current_rv, atm_iv_list, idx_recent = load_vol_data(idx_code, opt)

col_rv, col_cone = st.columns(2)

# ── 图1: 实际波动率 vs 近月ATM IV ──
with col_rv:
    st.markdown('<b>RV(20d) vs 近月ATM IV — 近60天</b>', unsafe_allow_html=True)
    fig_rv = go.Figure()
    # RV线
    if len(rv_series.dropna()) > 0:
        fig_rv.add_trace(go.Scatter(
            x=rv_series.index, y=rv_series.values * 100,
            mode="lines+markers", name="实际波动率(20d)",
            line=dict(color="#999", width=2, dash="dash"),
            marker=dict(size=4, color="#999")))
    # ATM IV线
    if atm_iv_list:
        atm_dates = pd.to_datetime([a["date"] for a in atm_iv_list])
        atm_vals = [a["atm_iv"] for a in atm_iv_list]
        atm_exp = atm_iv_list[-1]["expiry"] if atm_iv_list else "—"
        fig_rv.add_trace(go.Scatter(
            x=atm_dates, y=atm_vals,
            mode="lines+markers", name=f"近月ATM IV({atm_exp})",
            line=dict(color="#7b2ff7", width=2.5),
            marker=dict(size=4, color="#7b2ff7")))
    # 纵轴范围
    all_vals = []
    if len(rv_series.dropna()) > 0: all_vals += list(rv_series.dropna() * 100)
    if atm_iv_list: all_vals += [a["atm_iv"] for a in atm_iv_list]
    v_lo = max(0, min(all_vals) - 3) if all_vals else 0
    v_hi = max(all_vals) + 3 if all_vals else 40
    fig_rv.update_layout(
        template="plotly_white", height=360,
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        xaxis=dict(title="", showgrid=True, gridcolor="#f0f0f0", dtick=259200000,
                   tickformat="%m-%d"),
        yaxis=dict(title="年化波动率(%)", showgrid=True, gridcolor="#f0f0f0",
                   range=[v_lo, v_hi], ticksuffix="%", tickformat=".1f"),
        paper_bgcolor="#fff", plot_bgcolor="#fff", hovermode="x unified")
    st.plotly_chart(fig_rv, width='stretch', config={"displayModeBar": False})

# ── 图2: 波动率锥 ──
with col_cone:
    st.markdown('<b>波动率锥 — 全历史</b>', unsafe_allow_html=True)
    fig_cone = go.Figure()
    if vol_cone:
        windows = sorted(vol_cone.keys())
        ws = [w for w in windows if w in vol_cone]
        # 锥体分位线 — 从外到内颜色递减
        pct_configs = [
            ("max",  "#bbb",    "dot"),
            ("min",  "#bbb",    "dot"),
            ("p90",  "#c8c8c8", "dash"),
            ("p10",  "#c8c8c8", "dash"),
            ("p75",  "#d5d5d5", "dash"),
            ("p25",  "#d5d5d5", "dash"),
        ]
        for key, color, style in pct_configs:
            vals = [round(vol_cone[w][key] * 100, 2) for w in ws]
            if len(vals) == len(ws) and len(vals) > 1:
                fig_cone.add_trace(go.Scatter(
                    x=ws, y=vals, mode="lines",
                    name=key.upper(), line=dict(color=color, width=1, dash=style),
                    showlegend=False))
        # 填充区域
        for (lo_k, hi_k, fill_color) in [("p10", "p90", "rgba(160,160,160,0.20)"),
                                          ("p25", "p75", "rgba(160,160,160,0.30)")]:
            lo_vals = [round(vol_cone[w][lo_k] * 100, 2) for w in ws]
            hi_vals = [round(vol_cone[w][hi_k] * 100, 2) for w in ws]
            if lo_vals and hi_vals:
                fig_cone.add_trace(go.Scatter(
                    x=ws + ws[::-1], y=hi_vals + lo_vals[::-1],
                    fill="toself", fillcolor=fill_color, mode="none",
                    name=f"{lo_k}-{hi_k}", showlegend=False, hoverinfo="skip"))
        # 中位数线
        med_vals = [round(vol_cone[w]["p50"] * 100, 2) for w in ws]
        if med_vals:
            fig_cone.add_trace(go.Scatter(
                x=ws, y=med_vals, mode="lines+markers",
                name="中位数", line=dict(color="#444", width=2.2),
                marker=dict(size=5, color="#444"), showlegend=True))
        # 当前RV线
        if current_rv:
            crv_ws = sorted(current_rv.keys())
            crv_vals = [round(current_rv[w] * 100, 2) for w in crv_ws]
            fig_cone.add_trace(go.Scatter(
                x=crv_ws, y=crv_vals, mode="lines+markers",
                name="当前RV", line=dict(color="#e74c3c", width=2.8),
                marker=dict(size=8, color="#e74c3c", symbol="diamond")))
        # 当前ATM IV水平线
        if atm_iv_list:
            cur_atm = round(atm_iv_list[-1]["atm_iv"], 2)
            fig_cone.add_hline(y=cur_atm, line_dash="dot", line_color="#7b2ff7",
                line_width=2,
                annotation_text=f"ATM IV {cur_atm:.2f}%", annotation_position="top right",
                annotation_font=dict(size=11, color="#7b2ff7"))

        cone_max = round(max([vol_cone[w]["max"] * 100 for w in ws]), 2) if ws else 50
        fig_cone.update_layout(
            template="plotly_white", height=360,
            margin=dict(l=40, r=30, t=10, b=40),
            legend=dict(orientation="h", y=1.08, font=dict(size=10)),
            xaxis=dict(title="回溯窗口(天)", showgrid=True, gridcolor="#f0f0f0", tickvals=windows),
            yaxis=dict(title="年化波动率(%)", showgrid=True, gridcolor="#f0f0f0",
                       range=[0, cone_max * 1.15], ticksuffix="%", tickformat=".1f"),
            paper_bgcolor="#fff", plot_bgcolor="#fff", hovermode="x unified")
    else:
        fig_cone.update_layout(height=320)
        st.info("数据不足，无法生成波动率锥")
    st.plotly_chart(fig_cone, width='stretch', config={"displayModeBar": False})

# ── 图3&4: 近月ATM IV + 近月偏度 (近90天) ──
st.markdown('<b>近月ATM IV & 偏度 — 近90天</b>',unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_near_month_series(idx_code, opt_code):
    """加载近月ATM IV和偏度的90天时间序列"""
    conn = sqlite3.connect(cfg.DB_PATH)
    try:
        idx_df = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE code=? ORDER BY date",
            conn, params=(idx_code,))
        idx_df["date"] = pd.to_datetime(idx_df["date"])
        idx_map = {str(row["date"])[:10]: float(row["close"]) for _, row in idx_df.iterrows()}

        # 获取最近90个交易日的期权日期
        dates = pd.read_sql_query(
            "SELECT DISTINCT date FROM options_daily WHERE code=? ORDER BY date DESC LIMIT 90",
            conn, params=(opt_code,))
    finally:
        conn.close()

    atm_iv_list = []
    skew_list = []
    r = cfg.RISK_FREE_RATE
    conn2 = sqlite3.connect(cfg.DB_PATH)
    try:
        for _, dt_row in dates.iterrows():
            d = str(dt_row["date"])[:10]
            spot_p = idx_map.get(d)
            if spot_p is None:
                continue

            df = pd.read_sql_query(
                "SELECT expiry, strike, otype, close FROM options_daily WHERE code=? AND date=? AND close>0",
                conn2, params=(opt_code, d))
            if df.empty:
                continue

            # 近月（距到期≤2天则换到次近月，避免到期日IV失真）
            exps = sorted(df["expiry"].unique())
            near_exp = exps[0]
            ym0 = str(near_exp)
            T0 = max((datetime(2000+int(ym0[:2]),int(ym0[2:]),15)-datetime.strptime(d,"%Y-%m-%d")).days,1)/365
            if T0 * 365 <= 2 and len(exps) > 1:
                near_exp = exps[1]
            edf = df[df["expiry"] == near_exp].copy()
            if edf.empty:
                continue

            # ATM IV & 偏度 — 先算合成远期
            ym = str(near_exp)
            T = max((datetime(2000+int(ym[:2]),int(ym[2:]),15)-datetime.strptime(d,"%Y-%m-%d")).days,1)/365
            price_map = {}
            for _, row in edf.iterrows():
                price_map[(row["strike"], row["otype"])] = row["close"]
            fwd_map = {}
            for k in edf["strike"].unique():
                c = price_map.get((k, "C"))
                p = price_map.get((k, "P"))
                if c is not None and p is not None:
                    fwd_map[k] = k + (c - p) * np.exp(r * T)

            # ATM IV
            edf["dist"] = (edf["strike"] - spot_p).abs()
            atm_rows = edf.nsmallest(2, "dist")
            atm_ivs = []
            for _, ar in atm_rows.iterrows():
                F = fwd_map.get(ar["strike"], spot_p)
                iv_v = implied_vol(ar["close"], F, ar["strike"], T, r, ar["otype"])
                if not np.isnan(iv_v):
                    atm_ivs.append(iv_v * 100)
            if atm_ivs:
                atm_iv_list.append({"date": d, "atm_iv": round(np.mean(atm_ivs), 2), "expiry": near_exp})

            # 偏度 = IV(Put Δ=-0.25) - IV(Call Δ=+0.25)
            ec = edf[edf["otype"]=="C"]; ep = edf[edf["otype"]=="P"]
            if ec.empty or ep.empty:
                continue
            best_call, best_put = None, None
            for _, r_ in ec.iterrows():
                F = fwd_map.get(r_["strike"], spot_p)
                ivv = implied_vol(r_["close"], F, r_["strike"], T, r, "C")
                if np.isnan(ivv): continue
                d_ = bs_delta(F, r_["strike"], T, r, ivv, "C")
                if abs(d_ - 0.25) < 0.5:
                    if best_call is None or abs(d_ - 0.25) < abs(best_call[1] - 0.25):
                        best_call = (ivv, d_)
            for _, r_ in ep.iterrows():
                F = fwd_map.get(r_["strike"], spot_p)
                ivv = implied_vol(r_["close"], F, r_["strike"], T, r, "P")
                if np.isnan(ivv): continue
                d_ = bs_delta(F, r_["strike"], T, r, ivv, "P")
                if abs(d_ + 0.25) < 0.5:
                    if best_put is None or abs(d_ + 0.25) < abs(best_put[1] + 0.25):
                        best_put = (ivv, d_)
            if best_call is not None and best_put is not None:
                sk = round(best_put[0] - best_call[0], 4)
                skew_list.append({"date": d, "skew": sk, "expiry": near_exp})
    finally:
        conn2.close()

    return sorted(atm_iv_list, key=lambda x: x["date"]), sorted(skew_list, key=lambda x: x["date"])

atm_iv_90d, skew_90d = load_near_month_series(idx_code, opt)

col_nm1, col_nm2 = st.columns(2)

with col_nm1:
    st.markdown('<b>近月ATM IV (90天)</b>', unsafe_allow_html=True)
    fig_nm1 = go.Figure()
    if atm_iv_90d:
        dates = pd.to_datetime([a["date"] for a in atm_iv_90d])
        vals = [a["atm_iv"] for a in atm_iv_90d]
        fig_nm1.add_trace(go.Scatter(x=dates, y=vals, mode="lines+markers",
            name="ATM IV", line=dict(color="#7b2ff7", width=2.5), marker=dict(size=4, color="#7b2ff7")))
        for label, p in [("90%",90),("75%",75),("50%",50),("25%",25),("10%",10)]:
            pv = round(np.percentile(vals, p), 2)
            fig_nm1.add_hline(y=pv, line_dash="dash", line_color="#999", line_width=1,
                annotation_text=f"{label}={pv}", annotation_position="right",
                annotation_font=dict(size=9, color="#888"))
        rng = max(vals)-min(vals); pad = max(rng*0.08, 0.5)
        fig_nm1.update_layout(template="plotly_white", height=320,
            margin=dict(l=30,r=20,t=10,b=40),
            xaxis=dict(title="",showgrid=True,gridcolor="#f0f0f0",dtick=86400000*14,tickformat="%m-%d"),
            yaxis=dict(title="IV%",showgrid=True,gridcolor="#f0f0f0",range=[max(0,min(vals)-pad),max(vals)+pad]),
            paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified")
    else:
        fig_nm1.update_layout(height=320)
    st.plotly_chart(fig_nm1, width='stretch', config={"displayModeBar": False})

with col_nm2:
    st.markdown('<b>近月偏度 (90天)</b>', unsafe_allow_html=True)
    fig_nm2 = go.Figure()
    if skew_90d:
        dates = pd.to_datetime([a["date"] for a in skew_90d])
        vals = [a["skew"] for a in skew_90d]
        fig_nm2.add_trace(go.Scatter(x=dates, y=vals, mode="lines+markers",
            name="偏度", line=dict(color="#e67e22", width=2.5), marker=dict(size=4, color="#e67e22")))
        fig_nm2.add_hline(y=0, line_dash="dash", line_color="#ccc")
        for label, p in [("90%",90),("75%",75),("50%",50),("25%",25),("10%",10)]:
            pv = round(np.percentile(vals, p), 2)
            fig_nm2.add_hline(y=pv, line_dash="dash", line_color="#999", line_width=1,
                annotation_text=f"{label}={pv}", annotation_position="right",
                annotation_font=dict(size=9, color="#888"))
        rng = max(vals)-min(vals); pad = max(rng*0.08, 0.01)
        fig_nm2.update_layout(template="plotly_white", height=320,
            margin=dict(l=30,r=20,t=10,b=40),
            xaxis=dict(title="",showgrid=True,gridcolor="#f0f0f0",dtick=86400000*14,tickformat="%m-%d"),
            yaxis=dict(title="偏度",showgrid=True,gridcolor="#f0f0f0",range=[min(vals)-pad,max(vals)+pad]),
            paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified")
    else:
        fig_nm2.update_layout(height=320)
    st.plotly_chart(fig_nm2, width='stretch', config={"displayModeBar": False})

# ═══════════════════════════ 4. 合约IV微笑 ═══════════════════════════
st.markdown('<div class="section-title">📋 合约IV微笑</div>',unsafe_allow_html=True)
# 直接取期权到期月，不再依赖期货合约月份
smile_expiries = sorted(OD["expiry"].unique())
# 取前3个到期月，同时查对应期货价格做参考
smile_cols_data = []
for exp in smile_expiries:
    fut_match = FT[FT["code"].str[2:] == exp]
    fut_close = float(fut_match["close"].iloc[0]) if len(fut_match) > 0 else None
    smile_cols_data.append((exp, fut_close))
if smile_cols_data:
    # 每行两个图
    for i in range(0, len(smile_cols_data), 2):
        pair = smile_cols_data[i:i+2]
        ct_cols = st.columns(2)
        for j, (col, (ct_month, ct_close)) in enumerate(zip(ct_cols, pair)):
            with col:
                oct_df=OD[OD["expiry"]==ct_month]
                oc=oct_df[oct_df["otype"]=="C"].sort_values("strike")
                op=oct_df[oct_df["otype"]=="P"].sort_values("strike")
                # 合成期货下Call/Put同价IV应相等，每个行权价取均值
                smile_dict = {}
                for _, r in oct_df.iterrows():
                    if not pd.isna(r["iv"]) and r["iv"] > 0:
                        # 最低流动性过滤：无成交且持仓<10的合约不参与IV曲线
                        if int(r.get("volume",0)) == 0 and int(r.get("open_interest",0)) < 10:
                            continue
                        k = r["strike"]
                        smile_dict[k] = (smile_dict.get(k, (0,0))[0]+r["iv"], smile_dict.get(k, (0,0))[1]+1)
                smile_pts = [(k, v/c) for k,(v,c) in smile_dict.items() if c>0]
                smile_df = pd.DataFrame(smile_pts, columns=["strike","iv"]).sort_values("strike") if smile_pts else pd.DataFrame()
                # ATM IV
                atm_c = oc.iloc[(oc["strike"]-spot).abs().values.argmin()] if not oc.empty else None
                atm_p = op.iloc[(op["strike"]-spot).abs().values.argmin()] if not op.empty else None
                atm_iv_val = round((atm_c["iv"]+atm_p["iv"])/2,2) if (atm_c is not None and atm_p is not None and not pd.isna(atm_c["iv"]) and not pd.isna(atm_p["iv"])) else None
                # 前一日
                prev_smile_dict = {}
                if PD is not None and not PD.empty:
                    pdo = PD[PD["expiry"]==ct_month]
                    for _, r in pdo.iterrows():
                        if not pd.isna(r["iv"]) and r["iv"] > 0:
                            if int(r.get("volume",0)) == 0 and int(r.get("open_interest",0)) < 10:
                                continue
                            k = r["strike"]
                            prev_smile_dict[k] = (prev_smile_dict.get(k, (0,0))[0]+r["iv"], prev_smile_dict.get(k, (0,0))[1]+1)
                prev_smile_pts = [(k, v/c) for k,(v,c) in prev_smile_dict.items() if c>0]
                prev_smile_df = pd.DataFrame(prev_smile_pts, columns=["strike","iv"]).sort_values("strike") if prev_smile_pts else pd.DataFrame()
                # OI: 按Call/Put分开，浅色=当日总OI，深色=增减量(从零轴)
                call_oi_prev, put_oi_prev = {}, {}
                if PD is not None and not PD.empty:
                    pdo = PD[PD["expiry"]==ct_month]
                    for _, r in pdo[pdo["otype"]=="C"].iterrows(): call_oi_prev[r["strike"]] = int(r["open_interest"])
                    for _, r in pdo[pdo["otype"]=="P"].iterrows(): put_oi_prev[r["strike"]] = int(r["open_interest"])
                call_strikes = sorted(oc["strike"].unique())
                call_oi_now = [int(oc[oc["strike"]==k]["open_interest"].sum()) for k in call_strikes]
                call_delta  = [call_oi_now[i] - call_oi_prev.get(k,0) for i,k in enumerate(call_strikes)]
                put_strikes = sorted(op["strike"].unique())
                put_oi_now  = [int(op[op["strike"]==k]["open_interest"].sum()) for k in put_strikes]
                put_delta   = [put_oi_now[i] - put_oi_prev.get(k,0) for i,k in enumerate(put_strikes)]
                all_strikes = sorted(set(list(call_strikes)+list(put_strikes)))
                bar_w = max((max(all_strikes)-min(all_strikes))*0.006, 20) if len(all_strikes)>1 else 40

                dte = max((datetime(2000+int(ct_month[:2]),int(ct_month[2:]),15)-datetime.strptime(OLD,"%Y-%m-%d")).days,1)
                fut_str = f'{ct_close:.1f}' if ct_close is not None else '—'
                iv_str = f'{atm_iv_val:.2f}' if atm_iv_val is not None else '—'
                st.markdown(f'<b>{ct_month} {dte}天</b> <span style="color:#5b6abf">{fut_str}</span> <span style="color:#7b2ff7">{iv_str}</span>',unsafe_allow_html=True)
                
                fig_ct = go.Figure()
                # 今日散点
                if not smile_df.empty:
                    # 二次多项式拟合基准线
                    poly = smile_fit(smile_df["strike"].values, smile_df["iv"].values)
                    if poly:
                        sv_strikes = np.linspace(smile_df["strike"].min(), smile_df["strike"].max(), 100)
                        sv_iv = poly(sv_strikes)
                        fig_ct.add_trace(go.Scatter(x=sv_strikes, y=sv_iv, mode="lines",
                            name="拟合基准", line=dict(color="#7b2ff7", width=2.5, shape="spline"),
                            yaxis="y"))
                        # 在真实行权价位置标点
                        fig_ct.add_trace(go.Scatter(x=smile_df["strike"], y=poly(smile_df["strike"]),
                            mode="markers", name="行权价点", marker=dict(size=6, color="#7b2ff7", line=dict(width=1, color="#fff")),
                            showlegend=False, yaxis="y"))
                # 前一日 SVI
                if not prev_smile_df.empty:
                    poly_prev = smile_fit(prev_smile_df["strike"].values, prev_smile_df["iv"].values)
                    if poly_prev:
                        sv_iv_prev = poly_prev(sv_strikes)
                        fig_ct.add_trace(go.Scatter(x=sv_strikes, y=sv_iv_prev, mode="lines",
                            name="昨拟合基准", line=dict(color="#888", width=1.8, dash="dot", shape="spline"),
                            yaxis="y"))
                        fig_ct.add_trace(go.Scatter(x=prev_smile_df["strike"], y=poly_prev(prev_smile_df["strike"]),
                            mode="markers", marker=dict(size=6, color="#888", line=dict(width=1, color="#fff")),
                            showlegend=False, yaxis="y"))
                # Call OI: 浅红=当日 + 深红=增减
                fig_ct.add_trace(go.Bar(x=call_strikes,y=call_oi_now,name="Call OI",
                    marker_color="#f5b7b1",opacity=0.5,width=bar_w,offset=-bar_w/2,yaxis="y2"))
                fig_ct.add_trace(go.Bar(x=call_strikes,y=call_delta,name="Call增减",
                    marker_color="#e74c3c",opacity=0.85,width=bar_w,offset=-bar_w/2,yaxis="y2"))
                # Put OI: 浅绿=当日 + 深绿=增减
                fig_ct.add_trace(go.Bar(x=put_strikes,y=put_oi_now,name="Put OI",
                    marker_color="#a9dfbf",opacity=0.5,width=bar_w,offset=bar_w/2,yaxis="y2"))
                fig_ct.add_trace(go.Bar(x=put_strikes,y=put_delta,name="Put增减",
                    marker_color="#27ae60",opacity=0.85,width=bar_w,offset=bar_w/2,yaxis="y2"))
                fig_ct.add_vline(x=spot,line_dash="dash",line_color="#999")
                # 合成指数价格（实线）
                fwd_vals = oct_df["fwd"].dropna()
                if len(fwd_vals) > 0:
                    syn_idx = float(fwd_vals.mean())
                    fig_ct.add_vline(x=syn_idx, line_dash="solid", line_color="#e67e22",
                        annotation_text=f"合成{syn_idx:.1f}", annotation_position="top",
                        annotation_font=dict(size=9, color="#e67e22"))
                all_iv = list(smile_df["iv"]) if not smile_df.empty else []
                if not prev_smile_df.empty: all_iv += list(prev_smile_df["iv"])
                iv_lo = max(0, min(all_iv)-2) if all_iv else 15
                iv_hi = max(all_iv)+2 if all_iv else 35
                fig_ct.update_layout(template="plotly_white",height=300,margin=dict(l=30,r=50,t=10,b=40),
                    legend=dict(orientation="h",y=1.05,font=dict(size=8)),
                    xaxis=dict(title="行权价",showgrid=False),
                    yaxis=dict(title="IV%",showgrid=False,range=[iv_lo,iv_hi]),
                    yaxis2=dict(title="持仓量",showgrid=False,overlaying="y",side="right"),
                    paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified",
                    barmode="overlay")
                st.plotly_chart(fig_ct,width='stretch',config={"displayModeBar":False})

# ═══════════════════════════ 4.5 波动率曲面 ═══════════════════════════
st.markdown('<div class="section-title">🌐 波动率曲面</div>',unsafe_allow_html=True)
# 构建全合约IV矩阵
all_exps = sorted(OD["expiry"].unique())
all_strikes_surf = sorted(OD["strike"].unique())
# 过滤行权价范围
ref_surf = spot
surf_strikes = [k for k in all_strikes_surf if ref_surf*0.7 <= k <= ref_surf*1.3]
if len(all_exps) >= 2 and len(surf_strikes) >= 3:
    # IV热力图数据
    iv_matrix = []
    oi_matrix = []
    exp_labels_surf = []
    for exp in all_exps:
        edf = OD[OD["expiry"]==exp]
        dte = max((datetime(2000+int(exp[:2]),int(exp[2:]),15)-datetime.strptime(OLD,"%Y-%m-%d")).days,1)
        exp_labels_surf.append(f'{exp}({dte}d)')
        row_iv = []
        row_oi = []
        for k in surf_strikes:
            vals = edf[edf["strike"]==k]["iv"]
            oi_vals = edf[edf["strike"]==k]["open_interest"]
            row_iv.append(float(vals.mean()) if len(vals)>0 and not pd.isna(vals.mean()) else None)
            row_oi.append(int(oi_vals.sum()) if len(oi_vals)>0 else 0)
        iv_matrix.append(row_iv)
        oi_matrix.append(row_oi)
    
    col1_surf, col2_surf = st.columns(2)
    with col1_surf:
        st.markdown('<b>IV曲面(%)</b>',unsafe_allow_html=True)
        fig_surf = go.Figure(data=go.Heatmap(
            z=iv_matrix, x=surf_strikes, y=exp_labels_surf,
            colorscale='RdYlBu_r', zmin=10, zmax=40,
            hoverongaps=False,
            hovertemplate='行权价: %{x}<br>到期: %{y}<br>IV: %{z:.1f}%<extra></extra>'))
        fig_surf.update_layout(template="plotly_white", height=300,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(title="行权价", showgrid=False),
            yaxis=dict(title="到期月"),
            paper_bgcolor="#fff", plot_bgcolor="#fff")
        st.plotly_chart(fig_surf, width='stretch', config={"displayModeBar": False})
    
    with col2_surf:
        st.markdown('<b>持仓量分布(手)</b>',unsafe_allow_html=True)
        # OI变化颜色: 红色=OI集中
        fig_oi_surf = go.Figure(data=go.Heatmap(
            z=oi_matrix, x=surf_strikes, y=exp_labels_surf,
            colorscale='Reds', hoverongaps=False,
            hovertemplate='行权价: %{x}<br>到期: %{y}<br>OI: %{z}手<extra></extra>'))
        fig_oi_surf.update_layout(template="plotly_white", height=300,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(title="行权价", showgrid=False),
            yaxis=dict(title="到期月"),
            paper_bgcolor="#fff", plot_bgcolor="#fff")
        st.plotly_chart(fig_oi_surf, width='stretch', config={"displayModeBar": False})

# ═══════════════════════════ 5. 智能分析 & 异动提醒 ═══════════════════════════
st.markdown('<div class="section-title">🧠 智能分析 & 异动提醒</div>',unsafe_allow_html=True)

alerts = []
advice = []

# ── PCR 信号 ──
if pcr_vol > 1.0:
    alerts.append(("⚠️", f"PCR(Vol)={pcr_vol:.2f} > 1.0，看跌情绪浓厚，Put成交量超过Call"))
    advice.append("PCR偏高，市场偏向防守，短期谨慎看多")
elif pcr_vol < 0.5:
    alerts.append(("🔔", f"PCR(Vol)={pcr_vol:.2f} < 0.5，过度看涨，注意回调风险"))
    advice.append("PCR极低，市场情绪过热，关注冲高回落")

if pcr_oi > 0.9:
    alerts.append(("⚠️", f"PCR(OI)={pcr_oi:.2f} > 0.9，Put持仓累积，下方保护需求强"))
elif pcr_oi < 0.5:
    alerts.append(("📊", f"PCR(OI)={pcr_oi:.2f} < 0.5，Call持仓主导，市场偏乐观"))

# ── 基差信号 ──
if basis_v is not None and basis_pct is not None:
    if basis_pct < -1.0:
        alerts.append(("🔴", f"基差率={basis_pct:+.2f}%，深度贴水，市场对短期走势较悲观"))
        advice.append("期货深度贴水反映避险情绪，关注是否进一步扩大")
    elif basis_pct > 0.5:
        alerts.append(("🟢", f"基差率={basis_pct:+.2f}%，期货升水，市场情绪偏积极"))

# ── 偏度信号 ──
if skew_data:
    near_skew = skew_data[0]["skew"]
    prev_near_skew = prev_skew_map.get(skew_data[0]["expiry"])
    if near_skew > 0.15:
        alerts.append(("🔴", f"近月偏度={near_skew:.2f}，Put IV显著高于Call IV，市场恐慌/避险情绪明显"))
        advice.append("偏度极端偏高，Put溢价反映尾部风险担忧，可考虑卖出Put赚取溢价")
    elif near_skew < -0.05:
        alerts.append(("🟢", f"近月偏度={near_skew:.2f}，Call IV高于Put IV，市场追涨情绪"))
    if prev_near_skew is not None and abs(near_skew - prev_near_skew) > 0.05:
        direction = "上升" if near_skew > prev_near_skew else "下降"
        alerts.append(("⚡", f"偏度异动：日变化{abs(near_skew-prev_near_skew):.2f}（{direction}），关注情绪突变"))

# ── OI 异动 ──
oi_change_ratio = abs(tco+tpo - prev_call_oi - prev_put_oi) / max(1, prev_call_oi+prev_put_oi)
if oi_change_ratio > 0.15:
    alerts.append(("⚡", f"总持仓异动：日变化{oi_change_ratio*100:.1f}%，大幅增减仓"))
    if tco+tpo > prev_call_oi+prev_put_oi:
        advice.append("持仓量大幅增加，新资金入场，趋势可能延续")
    else:
        advice.append("持仓量大幅减少，资金离场，注意趋势转换")

# ── 最大痛点 ──
if max_pain_strike is not None:
    diff_pct = (max_pain_strike - spot) / spot * 100
    if diff_pct > 3:
        alerts.append(("📌", f"最大痛点{max_pain_strike}高于现价{diff_pct:.1f}%，到期前可能有向上引力"))
    elif diff_pct < -3:
        alerts.append(("📌", f"最大痛点{max_pain_strike}低于现价{abs(diff_pct):.1f}%，到期前可能有向下引力"))

# ── IV期限结构 ──
if atm_iv_data and len(atm_iv_data) >= 2:
    near_iv = atm_iv_data[0]["atm_iv"]
    far_iv = atm_iv_data[-1]["atm_iv"]
    if near_iv > far_iv + 3:
        alerts.append(("📉", f"IV期限倒挂：近月IV({near_iv:.1f}%) > 远月IV({far_iv:.1f}%)，短期波动预期高于长期"))
        advice.append("近月IV高于远月，市场预期短期有事件冲击，做多近月波动率需谨慎")
    elif far_iv > near_iv + 5:
        alerts.append(("📈", f"IV期限陡峭：远月IV({far_iv:.1f}%) >> 近月IV({near_iv:.1f}%)，远月风险溢价高"))

# ── 综合建议 ──
if not advice:
    advice.append(f"当前市场结构中性，{FUT_NAMES.get(u,'')}PCR={pcr_vol:.2f}、基差率={basis_pct:+.2f}%，观望为主")

# 渲染
if alerts:
    st.markdown("### 🚨 异动提醒")
    for icon, msg in alerts[:8]:
        st.markdown(f'<div style="background:#fff3cd;border-left:3px solid #e74c3c;padding:6px 10px;margin:3px 0;font-size:16px;border-radius:4px">{icon} {msg}</div>',unsafe_allow_html=True)

st.markdown("### 💡 投资建议")
for adv in advice:
    st.markdown(f'<div style="background:#e8f4fd;border-left:3px solid #5b6abf;padding:6px 10px;margin:3px 0;font-size:16px;border-radius:4px">📌 {adv}</div>',unsafe_allow_html=True)
