"""共享的数据定义和校验逻辑；训练、清洗、网页统一使用这里的字段规则。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET = "Grid_Status_Label"
CATEGORICAL = ["Charger_Type", "User_Type"]
NUMERIC = ["Start_Hour", "Energy_kWh", "Duration_Mins", "Overstay_Mins", "Battery_Temp_C"]
FEATURES = [*CATEGORICAL, *NUMERIC]
REQUIRED = ["Session_ID", *FEATURES, TARGET]
LABELS = ["Normal", "Overheated", "Grid_Surge"]
CHARGERS = ["AC", "DC"]
USERS = ["Student", "Staff", "Visitor"]
PEAK_RULE = "12 <= Start_Hour < 17（按开始小时；原作业的 12:00–17:00）"
SCHEMA_VERSION = "campus_sessions_v2"


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
    for col in ["Session_ID", *CATEGORICAL, TARGET]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    audit["missing_session_ids"] = int(df.Session_ID.isna().sum())
    df = df.dropna(subset=["Session_ID"])
    # 同一个会话编号只能出现一次；保留文件中第一次出现的记录。
    audit["duplicate_sessions_removed"] = int(df.Session_ID.duplicated().sum())
    df = df.drop_duplicates("Session_ID", keep="first").copy()
    audit["missing_before_imputation"] = {col: int(df[col].isna().sum()) for col in FEATURES}
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
        # 使用实际字段含义校验；不套用旧字典的环境温度范围，避免抹掉电池过热记录。
        if col == "Start_Hour":
            valid = df[col].between(0, 23) & df[col].eq(df[col].round())
        elif col == "Overstay_Mins":
            valid = df[col].ge(0)
        elif col == "Battery_Temp_C":
            valid = df[col].gt(-273.15)
        else:
            valid = df[col].gt(0)
        audit["invalid_numeric_values"][col] = int((original & ~valid).sum())
        df.loc[~valid, col] = np.nan
    for col, allowed in [("Charger_Type", CHARGERS), ("User_Type", USERS)]:
        unknown = set(df[col].dropna()) - set(allowed)
        if unknown:
            raise ValueError(f"{col} 存在未定义类别：{sorted(unknown)}，请核对数据说明。")
        # sklearn 兼容 numpy.nan；保留类别原意，不把 AC/DC 改成旧字典的名称。
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    audit["missing_targets"] = int(df[TARGET].isna().sum())
    if df.empty:
        raise ValueError("移除无编号记录后没有可用数据。")
    audit["retained_rows"] = len(df)
    return df.reset_index(drop=True), audit


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """充电功率是平均电量/时长。分组为分析用约定，不是实际能量转换效率。"""
    out = df.copy()
    # CSV 没有原始高峰标志。只按开始时间推导，不表示整段会话都发生在高峰期。
    out["Peak_Hour_Flag"] = ((out.Start_Hour >= 12) & (out.Start_Hour < 17)).astype(int)
    out["Power_kW"] = out.Energy_kWh / (out.Duration_Mins / 60.0)
    out["Energy_Efficiency_Group"] = pd.cut(
        out.Power_kW, bins=[-np.inf, 22, 50, np.inf],
        labels=["Low_power", "Medium_power", "High_power"], include_lowest=True,
    ).astype(str)
    return out
