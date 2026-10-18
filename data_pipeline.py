"""任务 1：读取 CSV、审计清洗、计算特征、导出分组汇总。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ev_core import ROOT, NUMERIC, engineer_features, file_hash, normalize_data


def build_dataset(source: Path, output: Path, demo: bool = False) -> pd.DataFrame:
    frame, audit = normalize_data(pd.read_csv(source))
    medians = {}
    for col in NUMERIC:
        # 二元标志不使用 0.5 这样的插补值，按类别众数填补。
        if frame[col].dropna().empty:
            raise ValueError(f"{col} 整列缺失或无效，无法从数据计算插补值。")
        value = frame[col].mode().iloc[0] if col == "Peak_Hour_Flag" else frame[col].median()
        medians[col] = float(value)
        frame[col] = frame[col].fillna(value)
    if frame.Charger_Type.dropna().empty:
        raise ValueError("Charger_Type 没有有效值，无法进行类别插补。")
    frame["Charger_Type"] = frame.Charger_Type.fillna(frame.Charger_Type.mode().iloc[0])
    # 保留作业额外提供的数值列，并用其中位数补齐；全空列保持空值并记入报告。
    extras = [c for c in frame.select_dtypes("number").columns if c not in NUMERIC]
    for col in extras:
        frame[col] = frame[col].fillna(frame[col].median())
    frame["Peak_Hour_Flag"] = frame.Peak_Hour_Flag.astype(int)
    frame = engineer_features(frame)
    output.mkdir(parents=True, exist_ok=True)
    report_dir = output / "reports"
    report_dir.mkdir(exist_ok=True)
    destination = output / "cleaned_ev_dataset.csv"
    frame.to_csv(destination, index=False)
    # 同时按充电桩种类和高峰标志聚合，便于比较不同组合。
    summary = frame.groupby(["Charger_Type", "Peak_Hour_Flag"], observed=True).agg(
        Session_Count=("Session_ID", "count"), Average_Energy_kWh=("Energy_kWh", "mean"),
        Average_Temp_C=("Ambient_Temp_C", "mean"), Average_Power_kW=("Power_kW", "mean"),
    ).reset_index()
    summary.to_csv(report_dir / "group_summary.csv", index=False)
    audit.update({"demo": demo, "source_sha256": file_hash(source),
                  "cleaned_sha256": file_hash(destination), "imputation_values": medians,
                  "remaining_missing": {k: int(v) for k, v in frame.isna().sum().items()}})
    (report_dir / "data_quality.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ev_campus_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = build_dataset(args.data, args.output_dir)
        print(f"清洗完成：{len(result)} 条记录，输出目录：{args.output_dir}")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"数据处理失败：{exc}\n")
