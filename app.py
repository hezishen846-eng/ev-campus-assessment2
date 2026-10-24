"""任务 3：三模块 Streamlit 门户，中文界面与中文代码注释。"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ev_core import CHARGERS, FEATURES, PEAK_RULE, ROOT, SCHEMA_VERSION, USERS, file_hash

st.set_page_config(page_title="EV Campus · 充电运营分析", page_icon="⚡", layout="wide")
sns.set_theme(style="whitegrid", palette="deep")


@st.cache_data
def load_data(path: str, version: str) -> pd.DataFrame:
    # version 是文件指纹，文件变化后自动使缓存失效。
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str, version: str):
    # 只加载本项目训练生成的文件，不提供任意 pickle 上传入口。
    return joblib.load(path)


def show_plot(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def dashboard(df: pd.DataFrame):
    st.title("充电运营概览")
    st.caption("按充电桩与时段筛选，查看校园充电使用情况。")
    st.caption(f"高峰标志由开始小时推导：{PEAK_RULE}。")
    chargers = st.multiselect("充电桩类型", CHARGERS, default=CHARGERS)
    peak = st.selectbox("充电时段", ["全部", "高峰", "非高峰"])
    filtered = df[df.Charger_Type.isin(chargers)].copy()
    if peak != "全部":
        filtered = filtered[filtered.Peak_Hour_Flag == int(peak == "高峰")]
    if filtered.empty:
        st.info("当前筛选条件下没有充电记录，请调整筛选条件。")
        return
    peak_records = filtered[filtered.Peak_Hour_Flag == 1]
    # 数据字典没有收入、电价字段。估算必须明确标注，不能伪装为实际收入。
    if "Revenue" in filtered and pd.to_numeric(peak_records.Revenue, errors="coerce").notna().all():
        revenue = pd.to_numeric(peak_records.Revenue).sum()
        revenue_label, suffix = "高峰收入（原始字段）", ""
        st.caption("收入使用 Revenue 字段，币种以数据来源为准。")
    else:
        rate = st.number_input("假设高峰电价（RM/kWh，仅供估算）", min_value=0.0, value=0.50, step=0.05)
        revenue = peak_records.Energy_kWh.sum() * rate
        revenue_label, suffix = "估算高峰收入", " RM"
        st.caption("CSV 没有收入或电价字段；此处为高峰开始会话的电量 × 假设电价，不代表实际营收或按时段拆分的账单。")
    cols = st.columns(4)
    cols[0].metric("充电会话", f"{len(filtered):,}")
    cols[1].metric("总供电量", f"{filtered.Energy_kWh.sum():,.1f} kWh")
    cols[2].metric("平均会话时长", f"{filtered.Duration_Mins.mean():.1f} 分钟")
    cols[3].metric(revenue_label, f"{revenue:,.2f}{suffix}")
    st.subheader("充电记录")
    st.dataframe(filtered, hide_index=True, width="stretch")
    st.download_button("下载筛选结果", filtered.to_csv(index=False).encode("utf-8-sig"), "filtered_sessions.csv", "text/csv")


def analytics(df: pd.DataFrame, folder: Path):
    st.title("可视化分析")
    st.caption("图表会随筛选更新；图内使用英文字段名，便于跨平台显示。")
    charger = st.selectbox("分析范围", ["全部", *CHARGERS])
    data = df if charger == "全部" else df[df.Charger_Type == charger]
    if data.empty:
        st.info("该类型暂无记录。")
        return
    tabs = st.tabs(["能量与温度", "时段对比", "类型分布", "模型评估"])
    with tabs[0]:
        color = st.selectbox("散点颜色分组", ["Grid_Status_Label", "Charger_Type", "Peak_Hour_Flag"])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.scatterplot(data=data, x="Battery_Temp_C", y="Energy_kWh", hue=color, alpha=0.65, ax=ax)
        ax.set(xlabel="Battery temperature (C)", ylabel="Energy (kWh)")
        show_plot(fig)
    with tabs[1]:
        measure = st.selectbox("对比指标", ["平均电量", "总电量", "会话数"])
        groups = data.groupby("Peak_Hour_Flag").Energy_kWh
        values = groups.mean() if measure == "平均电量" else groups.sum() if measure == "总电量" else groups.count()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Peak" if i else "Non-peak" for i in values.index], values.values, color=["#19788c", "#e0a34d"][:len(values)])
        ax.set_ylabel("Sessions" if measure == "会话数" else "Energy (kWh)")
        show_plot(fig)
    with tabs[2]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=data, x="Charger_Type", hue="Charger_Type", legend=False, ax=ax)
        ax.set(xlabel="Charger type", ylabel="Sessions")
        show_plot(fig)
        st.caption("Energy_Efficiency_Group 按平均充电功率分组；没有输入电能与电池储能数据，不能计算真实转换效率。")
    with tabs[3]:
        metrics = folder / "reports/model_comparison.csv"
        if not metrics.exists():
            st.info("请先运行训练流程生成评估报告。")
            return
        st.caption("按训练集交叉验证的宏平均 F1 选模型；accuracy 等列来自独立 20% 测试集。")
        comparison = pd.read_csv(metrics)
        st.dataframe(comparison, hide_index=True, width="stretch")
        name = st.selectbox("查看模型混淆矩阵", comparison.model.tolist())
        plot = folder / f"reports/confusion_{name}.png"
        if plot.exists():
            st.image(str(plot), width=650)
        report_path = folder / "reports/classification_reports.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())[name]
            st.dataframe(pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T, width="stretch")


def inference(folder: Path, df: pd.DataFrame):
    st.title("实时电网状态预测")
    st.caption("填写会话参数，模型将给出状态分类。电量、时长与超时是会话统计量，这不是充电开始前的预测。")
    model_file = folder / "ev_model.pkl"
    if not model_file.exists():
        st.info("尚无模型，请先运行数据处理与训练流程。")
        return
    bundle = load_model(str(model_file), file_hash(model_file))
    metadata = bundle["metadata"]
    if (metadata.get("demo") is not False or metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("features") != FEATURES
            or metadata.get("cleaned_sha256") != file_hash(folder / "cleaned_ev_dataset.csv")):
        st.error("数据与模型版本不一致，请重新运行完整流程。")
        return
    with st.sidebar.form("prediction_form"):
        st.subheader("本次充电参数")
        kind = st.selectbox("充电桩", CHARGERS)
        user = st.selectbox("用户类型", USERS)
        hour = st.number_input("开始小时（0–23）", min_value=0, max_value=23, value=9)
        energy = st.number_input("充电电量（kWh）", min_value=0.01, value=25.0)
        duration = st.number_input("连接时长（分钟）", min_value=1.0, value=120.0)
        overstay = st.number_input("超时停留（分钟）", min_value=0.0, value=0.0)
        # 温度范围取自本批数据，避免沿用环境温度的旧范围。
        low = float(df.Battery_Temp_C.min() // 1)
        high = float(df.Battery_Temp_C.max() // 1 + 1)
        temperature = st.slider("电池温度（°C）", low, high, min(high, max(low, 35.0)), 0.1)
        submit = st.form_submit_button("预测电网状态", type="primary")
    st.write(f"当前模型：**{metadata['selected_model']}**")
    if not submit:
        st.info("在左侧填写参数后，点击“预测电网状态”。")
        return
    row = pd.DataFrame([{"Charger_Type": kind, "User_Type": user, "Start_Hour": hour,
                         "Energy_kWh": energy, "Duration_Mins": duration,
                         "Overstay_Mins": overstay, "Battery_Temp_C": temperature}])[FEATURES]
    outside = [c for c in FEATURES if c in df.select_dtypes("number")
               and not df[c].min() <= row[c].iloc[0] <= df[c].max()]
    if outside:
        st.warning("这些输入超出了现有数据的观测范围，预测可能不可靠：" + "、".join(outside))
    pipeline = bundle["pipeline"]
    prediction = str(pipeline.predict(row)[0])
    messages = {"Normal": (st.success, "正常：模型预测为 Normal。"),
                "Overheated": (st.warning, "过热风险：模型预测为 Overheated。"),
                "Grid_Surge": (st.error, "电网突增风险：模型预测为 Grid_Surge。")}
    render, message = messages[prediction]
    render(message)
    st.metric("平均充电功率", f"{energy / (duration / 60):.2f} kW")
    proba = pipeline.predict_proba(row)[0]
    st.dataframe(pd.DataFrame({"类别": pipeline.classes_, "模型概率": proba}), hide_index=True, width="stretch")
    st.caption("分类概率未经校准，不等同于真实故障概率。模型使用本次提供的课程数据训练。")


def main():
    st.sidebar.title("⚡ EV Campus")
    st.sidebar.caption("智慧校园 · 充电运营门户")
    st.sidebar.caption("数据来源：课程提供的 ev_campus_dataset.csv")
    folder = ROOT
    source = folder / "cleaned_ev_dataset.csv"
    if not source.exists():
        st.info("尚未生成清洗数据，请放入 ev_campus_dataset.csv 后运行 run_pipeline.py。")
        st.stop()
    df = load_data(str(source), file_hash(source))
    page = st.sidebar.radio("功能导航", ["运营仪表盘", "可视化分析", "实时预测"])
    if page == "运营仪表盘":
        dashboard(df)
    elif page == "可视化分析":
        analytics(df, folder)
    else:
        inference(folder, df)


if __name__ == "__main__":
    main()
