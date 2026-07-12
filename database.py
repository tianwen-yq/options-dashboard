# -*- coding: utf-8 -*-
"""
SQLite 数据库层 — 建表、读写、查询
"""

import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH


def get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化所有表"""
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        -- 指数日线
        CREATE TABLE IF NOT EXISTS index_daily (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            code        TEXT    NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            amount      REAL,
            pct_change  REAL,
            UNIQUE(date, code)
        );

        -- 期货日线
        CREATE TABLE IF NOT EXISTS futures_daily (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    NOT NULL,
            code          TEXT    NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            volume        REAL,
            open_interest  REAL,
            pre_open_interest REAL,
            spot_close    REAL,
            basis         REAL,
            basis_rate    REAL,
            UNIQUE(date, code)
        );

        -- 期权逐合约日线
        CREATE TABLE IF NOT EXISTS options_daily (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    NOT NULL,
            code          TEXT    NOT NULL,
            instrument    TEXT    NOT NULL,
            expiry        TEXT    NOT NULL,
            strike        REAL   NOT NULL,
            otype         TEXT    NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            volume        REAL,
            open_interest  REAL,
            pre_open_interest REAL,
            moneyness     REAL,
            UNIQUE(date, instrument)
        );

        -- 期权日度汇总（PCR等）
        CREATE TABLE IF NOT EXISTS options_summary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL,
            code            TEXT    NOT NULL,
            pcr_volume      REAL,
            pcr_oi          REAL,
            total_call_vol  REAL,
            total_put_vol   REAL,
            total_call_oi   REAL,
            total_put_oi    REAL,
            UNIQUE(date, code)
        );

        -- 提醒日志
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            time        TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            level       TEXT    NOT NULL,
            message     TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_idx_date ON index_daily(date);
        CREATE INDEX IF NOT EXISTS idx_idx_code ON index_daily(code);
        CREATE INDEX IF NOT EXISTS idx_fut_date ON futures_daily(date);
        CREATE INDEX IF NOT EXISTS idx_fut_code ON futures_daily(code);
        CREATE INDEX IF NOT EXISTS idx_opt_date ON options_daily(date);
        CREATE INDEX IF NOT EXISTS idx_opt_code ON options_daily(code);
        CREATE INDEX IF NOT EXISTS idx_opt_instr ON options_daily(instrument);
    """)

    conn.commit()

    # ── 兼容性迁移：为旧表补充 open/high/low/close 列 ──
    _migrate_options_daily(conn)
    # ── 兼容性迁移：futures_daily 补充 pre_open_interest ──
    _migrate_futures_daily(conn)

    conn.close()


def _migrate_futures_daily(conn):
    """兼容旧表：补充 pre_open_interest 列"""
    cur = conn.cursor()
    existing = {r[1] for r in cur.execute("PRAGMA table_info(futures_daily)").fetchall()}
    if "pre_open_interest" not in existing:
        cur.execute("ALTER TABLE futures_daily ADD COLUMN pre_open_interest REAL")
    conn.commit()


def _migrate_options_daily(conn):
    """兼容旧表：补充 open/high/low/close 列（若缺失）"""
    cur = conn.cursor()
    existing = {r[1] for r in cur.execute("PRAGMA table_info(options_daily)").fetchall()}
    for col in ["open", "high", "low"]:
        if col not in existing:
            cur.execute(f"ALTER TABLE options_daily ADD COLUMN {col} REAL")
    # 旧列 price → close
    if "close" not in existing:
        if "price" in existing:
            cur.execute("ALTER TABLE options_daily ADD COLUMN close REAL")
            cur.execute("UPDATE options_daily SET close = price")
        else:
            cur.execute("ALTER TABLE options_daily ADD COLUMN close REAL")
    # 新增 pre_open_interest
    if "pre_open_interest" not in existing:
        cur.execute("ALTER TABLE options_daily ADD COLUMN pre_open_interest REAL")
    conn.commit()


# ═══════════════════════════ 写入 ═══════════════════════════

def insert_index_bars(df: pd.DataFrame):
    """写入指数日线（REPLACE 模式）"""
    if df.empty:
        return 0
    conn = get_conn()
    df = df.copy()
    df["date"] = df["date"].astype(str)
    rows = df[["date","code","open","high","low","close","volume","amount","pct_change"]].values.tolist()
    conn.executemany(
        "INSERT OR REPLACE INTO index_daily(date,code,open,high,low,close,volume,amount,pct_change) "
        "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return len(rows)


def insert_futures_bars(df: pd.DataFrame):
    """写入期货日线"""
    if df.empty:
        return 0
    conn = get_conn()
    df = df.copy()
    df["date"] = df["date"].astype(str)
    cols = ["date","code","open","high","low","close","volume","open_interest",
            "pre_open_interest","spot_close","basis","basis_rate"]
    rows = df[cols].values.tolist()
    conn.executemany(
        "INSERT OR REPLACE INTO futures_daily(date,code,open,high,low,close,volume,"
        "open_interest,pre_open_interest,spot_close,basis,basis_rate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return len(rows)


def insert_options_daily(date_str: str, code: str, df: pd.DataFrame):
    """写入期权逐合约日线（先删当天该品种，再批量插入）"""
    if df.empty:
        return 0
    conn = get_conn()
    conn.execute("DELETE FROM options_daily WHERE date=? AND code=?", (date_str, code))

    records = []
    for _, r in df.iterrows():
        records.append((
            date_str, code,
            str(r.get("instrument", "")),
            str(r.get("expiry", "")),
            float(r.get("strike", 0)),
            str(r.get("otype", "")),
            float(r.get("open", 0)),
            float(r.get("high", 0)),
            float(r.get("low", 0)),
            float(r.get("close", 0)),
            float(r.get("volume", 0)),
            float(r.get("open_interest", 0)),
            float(r.get("pre_open_interest", 0)),
            float(r.get("moneyness", 0)),
        ))
    conn.executemany(
        "INSERT INTO options_daily(date,code,instrument,expiry,strike,otype,"
        "open,high,low,close,volume,open_interest,pre_open_interest,moneyness) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
    conn.commit()
    conn.close()
    return len(records)


def insert_options_summary(date_str: str, code: str, summary: dict):
    """写入期权日度汇总（确保值转为 Python 原生类型，避免 numpy BLOB）"""
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO options_summary(date,code,pcr_volume,pcr_oi,"
        "total_call_vol,total_put_vol,total_call_oi,total_put_oi) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (date_str, code,
         float(summary.get("volume_pcr", 0)),
         float(summary.get("oi_pcr", 0)),
         int(summary.get("total_call_vol", 0)),
         int(summary.get("total_put_vol", 0)),
         int(summary.get("total_call_oi", 0)),
         int(summary.get("total_put_oi", 0))))
    conn.commit()
    conn.close()


def insert_alert(date_str: str, category: str, level: str, message: str):
    """写入提醒"""
    conn = get_conn()
    now = datetime.now().strftime("%H:%M:%S")
    conn.execute(
        "INSERT INTO alerts(date,time,category,level,message) VALUES (?,?,?,?,?)",
        (date_str, now, category, level, message))
    conn.commit()
    conn.close()


# ═══════════════════════════ 读取 ═══════════════════════════

def read_index_bars(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取指数日线"""
    conn = get_conn()
    sql = "SELECT * FROM index_daily WHERE code=?"
    params = [code]
    if start:
        sql += " AND date>=?"
        params.append(start)
    if end:
        sql += " AND date<=?"
        params.append(end)
    sql += " ORDER BY date"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_futures_bars(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取期货日线"""
    conn = get_conn()
    sql = "SELECT * FROM futures_daily WHERE code=?"
    params = [code]
    if start:
        sql += " AND date>=?"
        params.append(start)
    if end:
        sql += " AND date<=?"
        params.append(end)
    sql += " ORDER BY date"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_options_daily(code: str, date_str: str = None) -> pd.DataFrame:
    """读取期权日线（指定日期，默认最新）"""
    conn = get_conn()
    if date_str is None:
        date_str = _latest_date(conn, "options_daily", code)
        if date_str is None:
            conn.close()
            return pd.DataFrame()
    df = pd.read_sql_query(
        "SELECT * FROM options_daily WHERE code=? AND date=?",
        conn, params=(code, date_str))
    conn.close()
    return df


def read_options_summary(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取期权日度汇总"""
    conn = get_conn()
    sql = "SELECT * FROM options_summary WHERE code=?"
    params = [code]
    if start:
        sql += " AND date>=?"
        params.append(start)
    if end:
        sql += " AND date<=?"
        params.append(end)
    sql += " ORDER BY date"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_alerts(start: str = None, end: str = None, limit: int = 50) -> pd.DataFrame:
    """读取提醒"""
    conn = get_conn()
    sql = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if start:
        sql += " AND date>=?"
        params.append(start)
    if end:
        sql += " AND date<=?"
        params.append(end)
    sql += " ORDER BY date DESC, time DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_available_dates(table: str, code: str = None) -> list:
    """获取表中可用日期"""
    conn = get_conn()
    if code:
        rows = conn.execute(
            f"SELECT DISTINCT date FROM {table} WHERE code=? ORDER BY date DESC",
            (code,)).fetchall()
    else:
        rows = conn.execute(
            f"SELECT DISTINCT date FROM {table} ORDER BY date DESC").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ═══════════════════════════ 工具 ═══════════════════════════

def _latest_date(conn, table: str, code: str) -> str | None:
    row = conn.execute(
        f"SELECT MAX(date) FROM {table} WHERE code=?",
        (code,)).fetchone()
    return row[0] if row and row[0] else None


def db_stats() -> dict:
    """数据库统计"""
    conn = get_conn()
    tables = ["index_daily", "futures_daily", "options_daily", "options_summary", "alerts"]
    stats = {}
    for t in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        stats[t] = cnt
    conn.close()
    return stats


def clear_all_data() -> dict:
    """
    清空所有行情数据（指数 + 期货 + 期权）
    返回: 清除前后各表行数
    """
    conn = get_conn()
    tables = ["index_daily", "futures_daily", "options_daily", "options_summary"]
    before = {}
    for t in tables:
        before[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

    for t in tables:
        conn.execute(f"DELETE FROM {t}")

    conn.commit()

    after = {}
    for t in tables:
        after[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

    conn.close()
    return {"before": before, "after": after, "deleted": {k: before[k] - after[k] for k in before}}
