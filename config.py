# -*- coding: utf-8 -*-
"""
全局配置
"""

import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据库路径
DB_PATH = os.path.join(ROOT_DIR, "data", "options_analysis.db")

# 图表输出目录
CHART_DIR = os.path.join(ROOT_DIR, "charts")

# 提醒日志
ALERT_LOG = os.path.join(ROOT_DIR, "alerts.log")

# ── 指数代码 ──  (仅 IO/HO/MO 三大期权对应标的)
INDEX_CODES = {
    "000300.SH": "沪深300",
    "000852.SH": "中证1000",
    "000016.SH": "上证50",
}

# ── TQSDK 账号 ──
TQSDK_USER = os.environ.get("TQSDK_USER", "13419616215")
TQSDK_PWD  = os.environ.get("TQSDK_PWD", "yq19951028")

# ── TQSDK 指数符号映射 ──
TQ_INDEX_SYMBOLS = {
    "000300.SH": "SSE.000300",   # 沪深300
    "000852.SH": "SSE.000852",   # 中证1000
    "000016.SH": "SSE.000016",   # 上证50
}

# ── TQSDK 期货主连符号映射 ──
TQ_FUTURES_SYMBOLS = {
    "IF": "KQ.m@CFFEX.IF",
    "IM": "KQ.m@CFFEX.IM",
    "IH": "KQ.m@CFFEX.IH",
}

# ── 期货 → 现货指数 ──
FUTURES_TO_INDEX = {
    "IF": "000300.SH", "IM": "000852.SH", "IH": "000016.SH",
}

# ── 期权 → 期货品种 ──
OPTION_TO_FUTURES = {"IO": "IF", "HO": "IH", "MO": "IM"}

# ── 期权品种 ──
OPTION_CONFIG = {
    "IO": {"name": "沪深300股指期权", "multiplier": 100, "index": "000300.SH", "exchange": "CFFEX", "category": "index"},
    "HO": {"name": "上证50股指期权",   "multiplier": 100, "index": "000016.SH", "exchange": "CFFEX", "category": "index"},
    "MO": {"name": "中证1000股指期权", "multiplier": 100, "index": "000852.SH", "exchange": "CFFEX", "category": "index"},
}

# ── TQSDK 期权前缀映射 ──
TQ_OPTION_PREFIX = {"IO": "CFFEX.IO", "HO": "CFFEX.HO", "MO": "CFFEX.MO"}

# ── 均线参数 ──
MA_PERIODS = [5, 10, 20, 60, 120]

# ── 定价参数 ──
RISK_FREE_RATE = 0.012  # 无风险利率

# ── 异常检测阈值 ──
ALERT_THRESHOLDS = {
    "oi_change_ratio": 0.30,      # 持仓量单日变化超过30%
    "volume_spike":     3.0,      # 成交量超过20日均量3倍
    "pcr_change":       0.20,     # PCR单日变化超过20%
    "basis_extreme":    1.5,      # 基差率超过±1.5%
}

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
