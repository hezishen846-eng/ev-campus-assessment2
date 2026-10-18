"""用边界数据验证真实风险：错误输入、重复会话、泄漏和模型重新加载。"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from data_pipeline import build_dataset
from ev_core import FEATURES, NUMERIC, normalize_data
from make_demo import demo_frame
from train_model import train


def test_cleaning_and_audit(tmp_path):
    raw = demo_frame(120)
    raw.loc[0, "Duration_Mins"] = 0
    raw.loc[1, "Ambient_Temp_C"] = np.inf
    raw.loc[2, "Grid_Status_Label"] = "Grid Heavy"
    raw.loc[3, "Peak_Hour_Flag"] = 7
    source = tmp_path / "input.csv"
    raw.to_csv(source, index=False)
    clean = build_dataset(source, tmp_path)
    assert len(clean) == 120
    assert clean.Session_ID.is_unique
    assert clean[NUMERIC].notna().all().all()
    assert np.isfinite(clean.Power_kW).all()
    assert clean.Peak_Hour_Flag.isin([0, 1]).all()
    assert clean.loc[2, "Grid_Status_Label"] == "Grid_Surge"
    audit = json.loads((tmp_path / "reports/data_quality.json").read_text())
    assert audit["duplicate_sessions_removed"] == 8
    assert audit["invalid_numeric_values"]["Duration_Mins"] == 1


def test_missing_schema_rejected():
    with pytest.raises(ValueError, match="缺少必需字段"):
        normalize_data(pd.DataFrame({"x": [1]}))


def test_empty_column_rejected(tmp_path):
    raw = demo_frame(50)
    raw["Energy_kWh"] = np.nan
    path = tmp_path / "input.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="整列缺失"):
        build_dataset(path, tmp_path)


def test_unknown_label_rejected():
    raw = demo_frame(50)
    raw.loc[0, "Grid_Status_Label"] = "Mistyped_Label"
    with pytest.raises(ValueError, match="未定义的目标类别"):
        normalize_data(raw)


def test_training_no_leakage_and_reload(tmp_path):
    source = tmp_path / "input.csv"
    demo_frame(300).to_csv(source, index=False)
    build_dataset(source, tmp_path, demo=True)
    metadata = train(source, tmp_path, demo=True)
    assert metadata["train_rows"] == 240
    assert metadata["test_rows"] == 60
    assert set(metadata["train_session_ids"]).isdisjoint(metadata["test_session_ids"])
    loaded = joblib.load(tmp_path / "ev_model.pkl")
    raw, _ = normalize_data(pd.read_csv(source))
    train_rows = raw[raw.Session_ID.isin(metadata["train_session_ids"])]
    # 对照训练集的真实中位数，确保没有使用全数据的插补统计量。
    continuous = [c for c in NUMERIC if c != "Peak_Hour_Flag"]
    imputer = loaded["pipeline"].named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, train_rows[continuous].median().to_numpy())
    prediction = loaded["pipeline"].predict(raw[FEATURES].iloc[:5])
    assert len(prediction) == 5
    comparison = pd.read_csv(tmp_path / "reports/model_comparison.csv")
    assert len(comparison) == 3
    assert metadata["selected_model"] == comparison.iloc[0].model
    assert len(list((tmp_path / "reports").glob("confusion_*.png"))) == 3
