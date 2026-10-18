"""共享的数据定义和校验逻辑；训练、清洗、网页统一使用这里的字段规则。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET = "Grid_Status_Label"
NUMERIC = ["Energy_kWh", "Duration_Mins", "Peak_Hour_Flag", "Ambient_Temp_C"]
FEATURES = ["Charger_Type", *NUMERIC]
REQUIRED = ["Session_ID", *FEATURES, TARGET]
LABELS = ["Normal", "Overheated", "Grid_Surge"]
CHARGERS = ["AC_Fast", "DC_Ultra"]
RANGES = {"Energy_kWh": (5, 120), "Duration_Mins": (15, 300), "Ambient_Temp_C": (24, 42)}


def file_hash(path: str | Path) -> str:
    """记录输入文件指纹，避免把不同数据版本的模型与图表混用。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """只做不需要学习统计量的清洗；这里不计算中位数，避免测试集信息泄漏。"""
    df = frame.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError("CSV 存在重名列，请先修正列名。")
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"CSV 缺少必需字段：{', '.join(missing)}")
    audit = {"input_rows": len(df)}
    for col in ["Session_ID", "Charger_Type", TARGET]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    audit["missing_session_ids"] = int(df.Session_ID.isna().sum())
    df = df.dropna(subset=["Session_ID"])
    # 同一个会话编号只能出现一次；保留文件中第一次出现的记录。
    audit["duplicate_sessions_removed"] = int(df.Session_ID.duplicated().sum())
    df = df.drop_duplicates("Session_ID", keep="first").copy()
    # 原文存在 Grid Heavy / Grid_Surge 两种拼写，统一到任务 2 的目标名称。
    audit["label_aliases_normalized"] = int(df[TARGET].eq("Grid Heavy").sum())
    df[TARGET] = df[TARGET].replace({"Grid Heavy": "Grid_Surge"})
    unknown = set(df[TARGET].dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"发现未定义的目标类别：{sorted(unknown)}。请核对老师的数据说明。")
    audit["invalid_numeric_values"] = {}
    for col in NUMERIC:
        original = df[col].notna()
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = df[col].isin([0, 1]) if col == "Peak_Hour_Flag" else df[col].between(*RANGES[col])
        audit["invalid_numeric_values"][col] = int((original & ~valid).sum())
        df.loc[~valid, col] = np.nan
    audit["unknown_charger_values"] = int((df.Charger_Type.notna() & ~df.Charger_Type.isin(CHARGERS)).sum())
    df.loc[~df.Charger_Type.isin(CHARGERS), "Charger_Type"] = pd.NA
    # sklearn 兼容 numpy.nan；不把缺失目标猜成某一种状态。
    df["Charger_Type"] = df.Charger_Type.astype(object).where(df.Charger_Type.notna(), np.nan)
    audit["missing_targets"] = int(df[TARGET].isna().sum())
    if df.empty:
        raise ValueError("移除无编号记录后没有可用数据。")
    audit["retained_rows"] = len(df)
    return df.reset_index(drop=True), audit


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """充电功率是平均电量/时长。分组为分析用约定，不是实际能量转换效率。"""
    out = df.copy()
    out["Power_kW"] = out.Energy_kWh / (out.Duration_Mins / 60.0)
    out["Energy_Efficiency_Group"] = pd.cut(
        out.Power_kW, bins=[-np.inf, 22, 50, np.inf],
        labels=["Low_power", "Medium_power", "High_power"], include_lowest=True,
    ).astype(str)
    return out
