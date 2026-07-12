# -*- coding: utf-8 -*-
"""
每日数据更新脚本 — 模式1：最新快照
────────────────────────────────────
用途: 获取最新交易日的指数/期货/期权快照数据

两种数据更新模式：
  模式1 (日常): python run_daily_update.py     → 获取最新快照数据
  模式2 (回填): python 回填历史数据.py --start ... --end ... → 逐日补充历史

用法:
    python run_daily_update.py

流程:
    1. 初始化数据库（首次运行时建表）
    2. TQSDK 确认实际最近交易日
    3. 更新指数日线（增量）
    4. 更新期货日线（增量）
    5. 更新期权日线（当日最新 OHLC）
    6. 打印数据库统计
"""

import sys
import os

# 确保能找到同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db
from data_fetcher import daily_update_all

if __name__ == "__main__":
    print("=" * 60)
    print("  股指期权数据每日更新 (模式1: 最新快照)")
    print("=" * 60)

    # 初始化数据库
    init_db()
    print("✓ 数据库已就绪\n")

    # 执行全量更新
    daily_update_all()

    print("\n✅ 每日更新完成")
    print("  💡 如需回填历史数据，请运行: 回填历史数据.bat")
