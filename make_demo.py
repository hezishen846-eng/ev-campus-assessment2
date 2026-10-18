"""生成独立的合成演示数据，只用于验证软件功能，不能替代老师的正式数据。"""
import numpy as np
import pandas as pd

from ev_core import ROOT
from run_pipeline import run


def demo_frame(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    charger = rng.choice(["AC_Fast", "DC_Ultra"], n, p=[0.65, 0.35])
    duration = rng.uniform(45, 300, n)
    power = np.where(charger == "AC_Fast", rng.uniform(7.4, 22, n), rng.uniform(50, 110, n))
    energy = np.clip(power * duration / 60, 5, 120)
    temperature = rng.uniform(24, 42, n)
    peak = rng.integers(0, 2, n)
    # 标签由人为规则模拟，因此演示模型的高分不代表真实预测能力。
    status = np.where(temperature > 37, "Overheated",
                      np.where((peak == 1) & (energy / (duration / 60) > 32), "Grid_Surge", "Normal"))
    df = pd.DataFrame({"Session_ID": [f"DEMO-{i+1:04d}" for i in range(n)],
                       "Charger_Type": charger, "Energy_kWh": energy.round(2),
                       "Duration_Mins": duration.round(2), "Peak_Hour_Flag": peak,
                       "Ambient_Temp_C": temperature.round(2), "Grid_Status_Label": status})
    # 故意加入少量缺失值与重复记录，验证清洗流程确实工作。
    df.loc[rng.choice(n, max(1, n // 50), replace=False), "Energy_kWh"] = np.nan
    df.loc[rng.choice(n, max(1, n // 80), replace=False), "Ambient_Temp_C"] = np.nan
    return pd.concat([df, df.iloc[: min(8, n)]], ignore_index=True)


if __name__ == "__main__":
    folder = ROOT / "demo"
    folder.mkdir(exist_ok=True)
    source = folder / "synthetic_ev_dataset.csv"
    demo_frame().to_csv(source, index=False)
    run(source, folder, demo=True)
