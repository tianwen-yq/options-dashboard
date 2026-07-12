# -*- coding: utf-8 -*-
"""
数据获取模块 — 基于 TQSDK（天勤）
职责：
  - 每日更新指数/期货日线（全量/增量）
  - 每日获取最新期权快照数据（仅当天）
  - 历史期权数据补充（逐日回填）
  - 所有数据写入 SQLite
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from tqsdk import TqApi, TqAuth

import config as cfg
import database as db


# ╔══════════════════════════════════════════════════════════════╗
# ║  TQSDK 连接                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def _get_api() -> TqApi:
    """获取 TQSDK 连接"""
    return TqApi(auth=TqAuth(cfg.TQSDK_USER, cfg.TQSDK_PWD))


def get_latest_trading_day() -> str:
    """
    从 index_daily 表获取最近一个交易日日期
    不依赖系统时钟，数据里最新的一天就是最近交易日
    """
    conn = db.get_conn()
    row = conn.execute("SELECT MAX(date) FROM index_daily").fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return datetime.now().strftime("%Y-%m-%d")


def _resolve_trading_day(api: TqApi) -> str:
    """
    通过 TQSDK 确认实际最近交易日。
    从沪深300指数日线中提取最新有效日期，自动处理非交易日（周末/节假日）。
    如果当天是交易日则返回当天，否则返回最近一个交易日。
    失败时回退到 get_latest_trading_day()，并做周末修正。
    """
    try:
        kl = api.get_kline_serial("SSE.000300", 86400, data_length=10)
        api.wait_update(deadline=time.time() + 10)

        if kl is not None and len(kl) > 0:
            # 从最新到最早找第一根有效日线
            for i in range(len(kl) - 1, -1, -1):
                row = kl.iloc[i]
                close_val = float(row["close"]) if row.close == row.close else 0
                if close_val > 0:
                    ts = row["datetime"]
                    if ts == ts:
                        td = datetime.fromtimestamp(int(ts) / 1e9).strftime("%Y-%m-%d")
                        print(f"  TQSDK trading day: {td}")
                        return td
    except Exception as e:
        print(f"  TQSDK day detection failed: {e}")

    # 回退：优先数据库，再做周末修正确保不会返回非交易日
    fallback = get_latest_trading_day()
    fallback_dt = datetime.strptime(fallback, "%Y-%m-%d")
    wd = fallback_dt.weekday()
    if wd == 5:       # 周六 → 回退到周五
        fallback_dt -= timedelta(days=1)
    elif wd == 6:     # 周日 → 回退到周五
        fallback_dt -= timedelta(days=2)
    fallback = fallback_dt.strftime("%Y-%m-%d")
    print(f"  Fallback trading day: {fallback}")
    return fallback


def _get_trading_days(api: TqApi, start: str, end: str) -> list:
    """
    从沪深300指数 k 线中提取 start~end 区间内所有真实国内交易日。
    自动排除周末 + 节假日，只有 k 线中有有效数据的日期才纳入。
    """
    # 估算需要的 bar 数量：日期跨度 / 1.4 ≈ 交易日数，加缓冲
    days_span = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    data_len = max(int(days_span / 1.4) + 50, 200)

    kl = api.get_kline_serial("SSE.000300", 86400, data_length=data_len)
    api.wait_update(deadline=time.time() + 15)

    if kl is None or len(kl) == 0:
        # 回退：用 pd.date_range B 频（仅排除周末）
        print("   无法从 k 线提取交易日，回退到工作日历")
        date_range = pd.date_range(start=start, end=end, freq="B")
        return [d.strftime("%Y-%m-%d") for d in date_range]

    trading_days = []
    for i in range(len(kl)):
        row = kl.iloc[i]
        ts = row["datetime"]
        if ts != ts:
            continue
        close_val = float(row["close"]) if row.close == row.close else 0
        if close_val <= 0:
            continue
        d = datetime.fromtimestamp(int(ts) / 1e9).strftime("%Y-%m-%d")
        if start <= d <= end:
            trading_days.append(d)

    trading_days.sort()
    if trading_days:
        print(f"   K线提取交易日: {len(trading_days)} 天 ({trading_days[0]} ~ {trading_days[-1]})")
    else:
        print("   区间内无交易日")
    return trading_days


# ╔══════════════════════════════════════════════════════════════╗
# ║  指数日线                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def update_all_index_bars(start_date: str = None, end_date: str = None):
    """
    更新所有宽基指数日线到数据库
    start_date: 起始日期（默认 None=仅取最新，填入日期=回填起点）
    end_date:   结束日期（默认 None=到最近交易日，填入则精确控制区间）
    """
    api = _get_api()
    total = 0

    try:
        ref_day = end_date if end_date else _resolve_trading_day(api)

        for idx_code, name in cfg.INDEX_CODES.items():
            existing = db.read_index_bars(idx_code)
            if not existing.empty:
                last_date = existing["date"].max().strftime("%Y-%m-%d")
                if last_date >= ref_day:
                    print(f"  {name} ({idx_code}) 数据已是最新 -> 跳过")
                    continue
                fetch_start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                # DB 为空时：有 start_date 则全量回填，无则仅取最新一天
                fetch_start = start_date if start_date else ref_day

            print(f"  获取 {name} ({idx_code})  {fetch_start} -> {ref_day} ...", end=" ")

            try:
                bars = _fetch_single_index(api, idx_code, fetch_start, ref_day)
                if not bars.empty:
                    n = db.insert_index_bars(bars)
                    total += n
                    print(f" {n}条")
                else:
                    print(" 无数据")
            except Exception as e:
                print(f" {e}")
    finally:
        api.close()

    print(f"  指数更新完成，共 {total} 条\n")
    return total


def _fetch_single_index(api: TqApi, index_code: str, start: str, end: str) -> pd.DataFrame:
    """通过 TQSDK k线 获取单只指数日线"""
    tq_sym = cfg.TQ_INDEX_SYMBOLS.get(index_code)
    if not tq_sym:
        print(f"   未知 TQSDK 符号: {index_code}")
        return pd.DataFrame()

    # 获取日线（86400秒 = 1天）
    kl = api.get_kline_serial(tq_sym, 86400, data_length=3000)
    api.wait_update(deadline=time.time() + 15)

    if kl is None or len(kl) == 0:
        return pd.DataFrame()

    records = []
    for i in range(len(kl)):
        row = kl.iloc[i]
        ts = row["datetime"]
        if ts != ts:
            continue
        d = datetime.fromtimestamp(int(ts) / 1e9).strftime("%Y-%m-%d")
        if d < start or d > end:
            continue

        close_val = float(row["close"]) if row.close == row.close else 0
        if close_val <= 0:
            continue

        records.append({
            "date": d,
            "code": index_code,
            "open":  float(row["open"]) if row.open == row.open else close_val,
            "high":  float(row["high"]) if row.high == row.high else close_val,
            "low":   float(row["low"]) if row.low == row.low else close_val,
            "close": close_val,
            "volume": float(row["volume"]) if row.volume == row.volume else 0,
            "amount": 0.0,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["close"].pct_change() * 100
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  期货日线                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def update_all_futures_bars(start_date: str = None, end_date: str = None):
    """
    更新所有股指期货合约日线（IF/IH/IM 全部具体合约）
    start_date: 起始日期（默认 None=仅取最新，填入日期=回填起点）
    end_date:   结束日期（默认 None=到最近交易日，填入则精确控制区间）
    批量订阅所有合约 k 线，一次 wait_update 全部拉取
    """
    api = _get_api()
    total = 0

    try:
        ref_day = end_date if end_date else _resolve_trading_day(api)

        # 发现所有期货合约（按品种分组：IF / IH / IM）
        all_futures = _discover_futures_symbols(api)
        index_map = {}
        for underlying, idx_code in cfg.FUTURES_TO_INDEX.items():
            if idx_code:  # 商品期货无对应指数，跳过
                index_map[underlying] = db.read_index_bars(idx_code)

        # 收集所有需要获取的合约及其日期范围
        # 快照模式(start_date=None): 只取近月合约(最近3个月)
        # 回填模式(start_date给定): 取全部合约
        ref_ym = int(ref_day[:4]) * 100 + int(ref_day[5:7])  # 如 202606
        fetch_tasks = []
        for underlying in ["IF", "IH", "IM"]:
            symbols = all_futures.get(underlying, [])
            if not symbols:
                print(f"  {underlying}: 未发现合约 → 跳过")
                continue
            idx_bars = index_map.get(underlying)
            for sym in symbols:
                # 提取合约代码（去掉交易所前缀）
                code = sym
                for ex in ["CFFEX.", "SHFE.", "INE.", "DCE.", "CZCE."]:
                    if code.startswith(ex):
                        code = code[len(ex):]
                        break

                # 快照模式：只取近月合约(最近3个月)
                if start_date is None:
                    try:
                        month_part = code[-4:] if len(code) >= 4 else code
                        contract_ym = (2000 + int(month_part[:2])) * 100 + int(month_part[2:]) \
                            if len(month_part) == 4 and month_part.isdigit() else 0
                        if 0 < contract_ym < ref_ym - 3:
                            continue
                    except (ValueError, IndexError):
                        pass

                existing = db.read_futures_bars(code)
                if not existing.empty:
                    last_date = existing["date"].max().strftime("%Y-%m-%d")
                    if last_date >= ref_day:
                        continue
                    fetch_start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    fetch_start = start_date if start_date else ref_day
                if fetch_start > ref_day:
                    continue
                fetch_tasks.append((sym, code, fetch_start, idx_bars))

        # 打印统计
        if start_date is None:
            print(f"  快照模式: {len(fetch_tasks)} 个近月合约待更新")
        else:
            print(f"  回填模式: {len(fetch_tasks)} 个合约待获取")

        if not fetch_tasks:
            print("  所有合约已是最新")
            return 0

        print(f"  共 {len(fetch_tasks)} 个合约待获取，批量订阅中...")

        # 批量订阅所有合约 k 线
        klines = {}
        for tq_sym, code, fetch_start, idx_bars in fetch_tasks:
            klines[tq_sym] = api.get_kline_serial(tq_sym, 86400, data_length=3000)

        # 一次 wait_update 全部拉取
        print(f"  等待数据... (最多60秒)")
        api.wait_update(deadline=time.time() + 60)

        # 逐个提取数据
        for tq_sym, code, fetch_start, idx_bars in fetch_tasks:
            kl = klines.get(tq_sym)
            if kl is None or len(kl) == 0:
                continue
            try:
                records = []
                for i in range(len(kl)):
                    row = kl.iloc[i]
                    ts = row["datetime"]
                    if ts != ts:
                        continue
                    d = datetime.fromtimestamp(int(ts) / 1e9).strftime("%Y-%m-%d")
                    if d < fetch_start or d > ref_day:
                        continue
                    close_val = float(row["close"]) if row.close == row.close else 0
                    if close_val <= 0:
                        continue
                    oi_val = row["close_oi"]
                    oi = int(oi_val) if oi_val == oi_val else 0
                    pre_oi_val = row["open_oi"]
                    pre_oi = int(pre_oi_val) if pre_oi_val == pre_oi_val else 0
                    vol_val = row["volume"]
                    vol = int(vol_val) if vol_val == vol_val else 0
                    records.append({
                        "date": d, "code": code,
                        "open":  float(row["open"]) if row.open == row.open else close_val,
                        "high":  float(row["high"]) if row.high == row.high else close_val,
                        "low":   float(row["low"]) if row.low == row.low else close_val,
                        "close": close_val,
                        "volume": vol, "open_interest": oi,
                        "pre_open_interest": pre_oi,
                    })
                if records:
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                    # 合并基差
                    if idx_bars is not None and not idx_bars.empty:
                        spot = idx_bars[["date", "close"]].copy()
                        spot = spot.rename(columns={"close": "spot_close"})
                        spot["date"] = pd.to_datetime(spot["date"])
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.merge(spot, on="date", how="left")
                    else:
                        df["spot_close"] = np.nan
                    df["basis"] = df["close"] - df["spot_close"]
                    df["basis_rate"] = np.where(
                        df["spot_close"].notna() & (df["spot_close"] != 0),
                        (df["basis"] / df["spot_close"]) * 100, np.nan)
                    n = db.insert_futures_bars(df)
                    total += n
            except Exception as e:
                print(f"    {code}  {e}")

        print(f"  期货更新完成，共 {total} 条\n")
    finally:
        api.close()

    return total


def update_futures_history(start_date: str = None, end_date: str = None):
    """逐日回填期货日线（以TQSDK数据为准，有则存）"""
    api = _get_api()
    total = 0

    try:
        if start_date is None:
            print("  ❌ 请指定 --start 起始日期")
            return 0

        # 从 TQSDK 获取全部已知合约，提取月份用于过滤
        all_futures = _discover_futures_symbols(api)
        all_codes = []  # [(tq_sym, code, underlying, ym), ...]
        for underlying in ["IF", "IH", "IM"]:
            for tq_sym in all_futures.get(underlying, []):
                code = tq_sym
                for ex in ["CFFEX.", "SHFE.", "INE.", "DCE.", "CZCE."]:
                    if code.startswith(ex):
                        code = code[len(ex):]
                        break
                month_part = code[-4:] if len(code) >= 4 else ""
                ym = (2000 + int(month_part[:2])) * 100 + int(month_part[2:]) \
                    if len(month_part) == 4 and month_part.isdigit() else 0
                all_codes.append((tq_sym, code, underlying, ym))

        if end_date is None:
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        trading_days = _get_trading_days(api, start_date, end_date)

        index_map = {}
        for underlying, idx_code in cfg.FUTURES_TO_INDEX.items():
            if idx_code:
                index_map[underlying] = db.read_index_bars(idx_code)

        for d in trading_days:
            d_ym = int(d[:4]) * 100 + int(d[5:7])

            for underlying in ["IF", "IH", "IM"]:
                # 按品种过滤：ym >= d_ym，取前12个
                candidates = [(tq, cd, ym) for tq, cd, ul, ym in all_codes
                              if ul == underlying and ym >= d_ym]
                candidates.sort(key=lambda x: x[2])
                candidates = candidates[:12]

                idx_bars = index_map.get(underlying)
                for tq_sym, code, _ in candidates:
                    # 只检查 D 日是否已有数据
                    existing = db.read_futures_bars(code, start=d, end=d)
                    if not existing.empty:
                        continue

                    try:
                        kl = api.get_kline_serial(tq_sym, 86400, data_length=500)
                        api.wait_update(deadline=time.time() + 3)
                        if kl is None or len(kl) == 0:
                            continue

                        for i in range(len(kl)):
                            row = kl.iloc[i]
                            ts = row["datetime"]
                            if ts != ts:
                                continue
                            kd = datetime.fromtimestamp(int(ts) / 1e9).strftime("%Y-%m-%d")
                            if kd != d:
                                continue
                            close_val = float(row["close"]) if row.close == row.close else 0
                            if close_val <= 0:
                                continue
                            oi_val = row["close_oi"]
                            oi = int(oi_val) if oi_val == oi_val else 0
                            pre_oi_val = row["open_oi"]
                            pre_oi = int(pre_oi_val) if pre_oi_val == pre_oi_val else 0
                            vol_val = row["volume"]
                            vol = int(vol_val) if vol_val == vol_val else 0
                            spot_v = np.nan
                            if idx_bars is not None and not idx_bars.empty:
                                spot_row = idx_bars[idx_bars["date"] == pd.Timestamp(d)]
                                if not spot_row.empty:
                                    spot_v = float(spot_row["close"].iloc[0])
                            record = pd.DataFrame([{
                                "date": kd, "code": code,
                                "open":  float(row["open"]) if row.open == row.open else close_val,
                                "high":  float(row["high"]) if row.high == row.high else close_val,
                                "low":   float(row["low"]) if row.low == row.low else close_val,
                                "close": close_val,
                                "volume": vol, "open_interest": oi,
                                "pre_open_interest": pre_oi,
                                "spot_close": spot_v,
                                "basis": close_val - spot_v if not np.isnan(spot_v) else np.nan,
                                "basis_rate": ((close_val - spot_v) / spot_v * 100) if not np.isnan(spot_v) and spot_v != 0 else np.nan,
                            }])
                            record["date"] = pd.to_datetime(record["date"])
                            db.insert_futures_bars(record)
                            total += 1
                            break
                    except Exception:
                        pass

            if total:
                print(f"  {d}: 累计 {total} 条", end="\r")

        print(f"\n  期货历史回填完成，共 {total} 条")
    finally:
        api.close()
    return total


# ╔══════════════════════════════════════════════════════════════╗
# ║  期权 — 最新快照（日常更新）                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def update_options_today(code: str = None):
    """
    获取最新交易日期权快照数据并写入数据库
    完全参照 fetch_io_options.py 的模式：
      query_quotes() → get_quote_list() → wait_update()
    Quote 字段: open / highest / lowest / last_price / volume / open_interest / pre_open_interest

    交易日确认：通过 TQSDK 指数 k 线确认实际最近交易日

    code: 'IO'/'HO'/'MO' 或 None（全部）
    """
    api = _get_api()

    try:
        # ── 确认实际交易日 ──
        ref_day = _resolve_trading_day(api)
        codes = [code] if code else list(cfg.OPTION_CONFIG.keys())

        for oc in codes:
            oc_cfg = cfg.OPTION_CONFIG.get(oc)
            if not oc_cfg:
                continue

            # 检查该品种该交易日是否已有数据
            existing = db.read_options_daily(oc, ref_day)
            if not existing.empty:
                print(f"  {oc} {ref_day} 数据已存在 -> 跳过")
                continue

            print(f"\n{'='*50}")
            print(f"  {oc_cfg['name']} ({oc})  交易日: {ref_day}")
            print(f"{'='*50}")

            # 1) query_quotes() 发现所有活跃期权合约
            symbols = _discover_option_symbols(api, oc)
            # 用当日最早期货合约月份过滤，去掉已过期期权
            symbols = _filter_options_by_futures(symbols, oc, ref_day)
            if not symbols:
                print(f"  未发现 {oc} 合约 → 跳过")
                continue

            months = sorted(set(_parse_option_symbol(s)["month"] for s in symbols
                                if _parse_option_symbol(s) is not None))
            print(f"  活跃月份: {months}  合约数: {len(symbols)}")

            # 2) get_quote_list() 批量订阅报价
            #    参照 fetch_io_options.py 第三步
            quotes = api.get_quote_list(symbols)

            # 3) wait_update() 等待数据推送
            #    参照 fetch_io_options.py 第四步
            api.wait_update(deadline=time.time() + 15)

            # 获取当日指数收盘价用于 moneyness 计算
            idx_code = cfg.OPTION_CONFIG[oc].get("index")
            spot_now = _get_index_close(idx_code, ref_day) if idx_code else 0

            # 4) 提取报价字段
            #    open / highest / lowest / last_price / volume / open_interest / pre_open_interest
            rows = []
            for q in quotes:
                sym = q.instrument_id
                info = _parse_option_symbol(sym)
                if not info:
                    continue

                # NaN 检查：x == x 模式（参照参考代码）
                lp = float(q.last_price) if q.last_price == q.last_price else 0
                op = float(q.open) if q.open == q.open else 0
                hi = float(q.highest) if q.highest == q.highest else 0
                lo = float(q.lowest) if q.lowest == q.lowest else 0
                oi = int(q.open_interest) if q.open_interest == q.open_interest else 0
                pre_oi = int(q.pre_open_interest) if q.pre_open_interest == q.pre_open_interest else 0
                vol = int(q.volume) if q.volume == q.volume else 0

                if oi > 0 or lp > 0:
                    moneyness = info["strike"] / spot_now if spot_now > 0 else 0

                    rows.append({
                        "instrument": sym,
                        "expiry": info["month"],
                        "strike": info["strike"],
                        "otype": info["direction"],
                        "open": op if op > 0 else lp,
                        "high": hi if hi > 0 else lp,
                        "low": lo if lo > 0 else lp,
                        "close": lp,
                        "volume": vol,
                        "open_interest": oi,
                        "pre_open_interest": pre_oi,
                        "moneyness": moneyness,
                    })

            if not rows:
                print(f"  无有效数据 → 跳过")
                continue

            df = pd.DataFrame(rows)
            print(f"  有效合约: {len(df)} 个")

            # 5) 写入 options_daily
            n = db.insert_options_daily(ref_day, oc, df)
            print(f"  写入 options_daily: {n} 条")

            # 6) 计算汇总 PCR
            calls = df[df["otype"] == "C"]
            puts  = df[df["otype"] == "P"]
            tcv = calls["volume"].sum()
            tpv = puts["volume"].sum()
            tco = calls["open_interest"].sum()
            tpo = puts["open_interest"].sum()

            summary = {
                "volume_pcr":      round(tpv / tcv, 4) if tcv > 0 else 0,
                "oi_pcr":          round(tpo / tco, 4) if tco > 0 else 0,
                "total_call_vol":  round(tcv, 0),
                "total_put_vol":   round(tpv, 0),
                "total_call_oi":   round(tco, 0),
                "total_put_oi":    round(tpo, 0),
            }
            db.insert_options_summary(ref_day, oc, summary)
            print(f"  PCR_vol={summary['volume_pcr']:.3f}  PCR_oi={summary['oi_pcr']:.3f}")

    finally:
        api.close()

    print("\n期权更新完成\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  期权 — 历史数据补充（逐日回填）                              ║
# ╚══════════════════════════════════════════════════════════════╝

def update_options_history(code: str = None, target_date: str = None,
                           start_date: str = None,
                           end_date: str = None):
    """
    逐日补充历史期权数据（适用于本地数据库为空的情况）
    使用 TQSDK get_kline_serial 批量获取指定日期的 OHLC + 成交量 + 持仓量

    code:        'IO'/'HO'/'MO' 或 None（全部）
    target_date: 指定单日，如 "2026-06-25"
    start_date:  批量回填起始日期（target_date 为 None 时必填）
    end_date:    批量回填结束日期；为 None 则用未来日期兜底，K线自动截断

    优先级: target_date > (start_date + end_date)
    """
    api = _get_api()
    codes = [code] if code else list(cfg.OPTION_CONFIG.keys())

    try:
        if target_date:
            dates_to_fetch = [target_date]
        else:
            if start_date is None:
                print("  ❌ 请指定 --start 起始日期")
                return
            # 从沪深300 K线提取真实交易日（自动排除周末+节假日）
            # end_date=None 时用未来日期兜底，K线会自动截断到最近交易日
            if end_date is None:
                end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            dates_to_fetch = _get_trading_days(api, start_date, end_date)

        for oc in codes:
            oc_cfg = cfg.OPTION_CONFIG.get(oc)
            if not oc_cfg:
                continue

            print(f"\n{'='*50}")
            print(f"  {oc_cfg['name']} ({oc}) 历史数据补充")
            print(f"{'='*50}")

            # 发现全部期权合约（不做过滤，每天单独过滤）
            all_symbols = _discover_option_symbols(api, oc)
            if not all_symbols:
                print(f"  未发现 {oc} 合约 → 跳过")
                continue

            total_records = 0
            idx_code_hist = cfg.FUTURES_TO_INDEX.get(cfg.OPTION_TO_FUTURES.get(oc))
            for d in dates_to_fetch:
                # 检查是否已有数据
                existing = db.read_options_daily(oc, d)
                if not existing.empty:
                    continue

                # 当天根据活跃期货合约过滤期权
                symbols = _filter_options_by_futures(all_symbols, oc, d)
                if not symbols:
                    continue

                spot_hist = _get_index_close(idx_code_hist, d)

                # 当天重新订阅 k 线（合约到期/新上市自动适配）
                klines = {}
                for sym in symbols:
                    klines[sym] = api.get_kline_serial(sym, 86400, data_length=200)
                api.wait_update(deadline=time.time() + 30)

                # 从 k 线中提取目标日期的数据
                day_rows = []
                for sym in symbols:
                    kl = klines.get(sym)
                    if kl is None or len(kl) == 0:
                        continue

                    info = _parse_option_symbol(sym)
                    if not info:
                        continue

                    for i in range(len(kl)):
                        row = kl.iloc[i]
                        ts = row["datetime"]
                        # NaN 检查：x == x 模式（参照 test_kline_hist.py）
                        if ts != ts:
                            continue
                        row_date = datetime.fromtimestamp(int(ts) / 1e9).strftime("%Y-%m-%d")
                        if row_date == d:
                            close_val = float(row["close"]) if row.close == row.close else 0
                            if close_val > 0:
                                oi_val = row["close_oi"]
                                oi = int(oi_val) if oi_val == oi_val else 0
                                pre_oi_val = row["open_oi"]
                                pre_oi = int(pre_oi_val) if pre_oi_val == pre_oi_val else 0
                                vol_val = row["volume"]
                                vol = int(vol_val) if vol_val == vol_val else 0
                                open_val  = float(row["open"]) if row.open == row.open else close_val
                                high_val  = float(row["high"]) if row.high == row.high else close_val
                                low_val   = float(row["low"]) if row.low == row.low else close_val
                                day_rows.append({
                                    "instrument": sym,
                                    "expiry": info["month"],
                                    "strike": info["strike"],
                                    "otype": info["direction"],
                                    "open": open_val,
                                    "high": high_val,
                                    "low": low_val,
                                    "close": close_val,
                                    "volume": vol,
                                    "open_interest": oi,
                                    "pre_open_interest": pre_oi,
                                    "moneyness": info["strike"] / spot_hist if spot_hist > 0 else 0,
                                })
                            break

                if day_rows:
                    df = pd.DataFrame(day_rows)
                    n = db.insert_options_daily(d, oc, df)
                    total_records += n

                    # 计算 PCR
                    calls = df[df["otype"] == "C"]
                    puts  = df[df["otype"] == "P"]
                    tcv = calls["volume"].sum()
                    tpv = puts["volume"].sum()
                    tco = calls["open_interest"].sum()
                    tpo = puts["open_interest"].sum()
                    summary = {
                        "volume_pcr":      round(tpv / tcv, 4) if tcv > 0 else 0,
                        "oi_pcr":          round(tpo / tco, 4) if tco > 0 else 0,
                        "total_call_vol":  round(tcv, 0),
                        "total_put_vol":   round(tpv, 0),
                        "total_call_oi":   round(tco, 0),
                        "total_put_oi":    round(tpo, 0),
                    }
                    db.insert_options_summary(d, oc, summary)

                if len(dates_to_fetch) > 1:
                    print(f"    {d}: {len(day_rows)}条  (累计 {total_records})")

            print(f"  {oc} 历史补充完成，共 {total_records} 条")

    finally:
        api.close()

    print("\n历史期权数据补充完成\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  期权 — 工具函数                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def _get_index_close(idx_code: str, date_str: str) -> float:
    """从数据库获取指定日期指数收盘价，查不到返回0"""
    if not idx_code:
        return 0.0
    conn = db.get_conn()
    row = conn.execute(
        "SELECT close FROM index_daily WHERE code=? AND date=?",
        (idx_code, date_str)
    ).fetchone()
    conn.close()
    return float(row[0]) if row and row[0] else 0.0


def _filter_options_by_futures(symbols: list, oc: str, date_str: str = None) -> list:
    """
    根据日期过滤期权合约：只保留到期月 >= D日月份的合约。
    和期货过滤逻辑一致，期权月份比期货多，所以直接按日期过滤即可。
    """
    if date_str is None:
        return symbols
    d_ym = int(date_str[:4]) * 100 + int(date_str[5:7])

    filtered = []
    for s in symbols:
        info = _parse_option_symbol(s)
        if not info:
            continue
        month_str = info.get("month", "")
        if len(month_str) == 4 and month_str.isdigit():
            ym = (2000 + int(month_str[:2])) * 100 + int(month_str[2:])
            if ym >= d_ym:
                filtered.append(s)

    removed = len(symbols) - len(filtered)
    print(f"  期权月份过滤: D月={d_ym}, {len(symbols)} → {len(filtered)} 个合约"
          + (f" (过滤掉 <{d_ym} 的 {removed} 个)" if removed > 0 else ""))
    return filtered


def _discover_option_symbols(api: TqApi, option_code: str) -> list:
    """通过 query_quotes 发现 CFFEX 活跃期权合约"""
    prefix = cfg.TQ_OPTION_PREFIX.get(option_code)
    if not prefix:
        return []
    all_inst = api.query_quotes()
    symbols = sorted([s for s in all_inst if s.startswith(prefix)])
    return symbols


def _parse_option_symbol(symbol: str) -> dict:
    """
    解析 TQSDK CFFEX 期权合约符号
    例: "CFFEX.IO2607-C-4200" → {"option_code":"IO","month":"2607","direction":"C","strike":4200}
    """
    try:
        rest = symbol.replace("CFFEX.", "")
        parts = rest.split("-")
        if len(parts) >= 3:
            return {"option_code": rest[:2], "month": parts[0][2:], "direction": parts[1], "strike": int(parts[2])}
        return None
    except Exception:
        return None


def _discover_futures_symbols(api: TqApi) -> dict:
    """发现 CFFEX 股指期货合约（IF/IH/IM），排除主连"""
    all_inst = api.query_quotes()
    result = {}
    for underlying in ["IF", "IH", "IM"]:
        prefix = f"CFFEX.{underlying}"
        result[underlying] = sorted([s for s in all_inst if s.startswith(prefix) and "KQ." not in s])
    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║  一键全量更新                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def daily_update_all():
    """每日一键更新：指数 → 期货 → 期权"""
    print("=" * 60)
    print(f"  开始每日数据更新  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1/3] 更新指数日线")
    update_all_index_bars()

    print("[2/3] 更新期货日线")
    update_all_futures_bars()

    print("[3/3] 更新期权日线")
    update_options_today()

    # 统计
    stats = db.db_stats()
    print("=" * 60)
    print("  数据库统计:")
    for t, c in stats.items():
        print(f"    {t}: {c} 条")
    print("=" * 60)
