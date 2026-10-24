"""仅供自动化测试的边界样本；不会导出或作为网页、正式模型的数据。"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_frame():
    def make(n=120):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "Session_ID": [f"TEST-{i}" for i in range(n)],
            "Charger_Type": rng.choice(["AC", "DC"], n),
            "User_Type": rng.choice(["Student", "Staff", "Visitor"], n),
            "Start_Hour": rng.integers(0, 24, n),
            "Energy_kWh": rng.uniform(3, 50, n),
            "Duration_Mins": rng.uniform(15, 215, n),
            "Overstay_Mins": rng.integers(0, 180, n),
            "Battery_Temp_C": rng.uniform(17, 57, n),
            "Grid_Status_Label": np.resize(["Normal", "Overheated", "Grid_Surge"], n),
        })
        frame.loc[0, "Energy_kWh"] = np.nan
        return pd.concat([frame, frame.iloc[:8]], ignore_index=True)
    return make
