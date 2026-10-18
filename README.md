# EV Campus：校园充电数据分析与机器学习

Assessment 2（ITS72604）对应的 Python 项目。代码、关键设计说明和操作指引均带中文注释。

## 当前交付状态

- 已实现数据清洗、特征工程、分组分析、三种分类模型、评估图表和三模块 Streamlit 门户。
- 已配置独立 `.venv`，并提供三个 PyCharm 运行配置。
- 当前没有收到老师的 `ev_campus_dataset.csv`。`demo/` 中是明确标记的**合成演示数据与模型**，不是正式实验结果。演示数据中的标签来自人为规则，模型高分不能作为真实预测能力的证据。
- 正式的 `cleaned_ev_dataset.csv`、`ev_model.pkl` 和 `reports/` 会在提供真实 CSV 后生成在项目根目录。
- GitHub / Streamlit Cloud 的实际发布、5 分钟视频的录制与上传尚未进行；准备步骤见 `docs/部署与演示.md`。

## 在 PyCharm 中运行

项目已打开，并使用本项目 `.venv/bin/python`（Python 3.12）作为解释器。

右上角有三个运行配置：

1. **01 正式数据处理与训练**：先把老师的 CSV 放到项目根目录，命名为 `ev_campus_dataset.csv`，再运行。没有文件时会明确提示缺少数据。
2. **02 启动充电分析网页**：启动 Streamlit，按运行窗口中的链接打开浏览器。通常为 `http://127.0.0.1:8501`。
3. **03 生成演示数据**：重新生成 `demo/` 数据和演示模型。不会覆盖项目根目录的正式产物。

`app.py` 是 Streamlit 入口，使用配置 02 启动；普通的“运行 app.py”不是网页启动方式。

如果 8501 端口被当前预览占用，先停止旧预览，或在运行配置参数末尾加入 `--server.port 8502`。

## 在终端运行

以下命令在本项目目录执行。复制项目到新电脑时，重新创建虚拟环境，不复制 `.venv`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# 正式处理：真实 CSV 放入项目根目录后运行
python run_pipeline.py

# 或指定原始 CSV 的实际位置
python run_pipeline.py --data "/完整路径/ev_campus_dataset.csv"

# 单独生成演示数据，不当作正式作业数据
python make_demo.py

# 启动网页
python -m streamlit run app.py --server.address 127.0.0.1

# 功能测试（网页测试需要已生成 demo/）
python -m pytest -q
```

## 文件与评分要求的对应关系

| 任务 | 文件 | 完成内容 |
|---|---|---|
| 任务 1：6 分 | `data_pipeline.py`、`ev_core.py` | 校验、缺失值处理、去重、功率、分组 CSV、清洗审计 |
| 任务 2：9 分 | `train_model.py` | 80/20 分层划分、独热编码、随机森林/决策树/KNN 对比、四项指标、混淆矩阵、模型持久化 |
| 任务 3：9 分 | `app.py` | KPI 仪表盘、交互图表、实时预测，侧边栏导航 |
| 任务 4：6 分 | `.streamlit/`、`requirements.txt`、`docs/部署与演示.md` | 统一界面、依赖配置、部署步骤与视频提纲；实际发布及录制待完成 |
| 验证 | `tests/` | 数据异常、去重、无泄漏插补、模型重新加载、筛选和实时预测 |

## 数据处理约定

1. 必须有 `Session_ID`、`Charger_Type`、`Energy_kWh`、`Duration_Mins`、`Peak_Hour_Flag`、`Ambient_Temp_C`、`Grid_Status_Label` 七列。额外列会保留。
2. 无会话编号的记录删除；重复编号保留第一条；删除数量记录在审计报告中。
3. 超出题目字典范围、非数值、正负无穷大的核心数值转为缺失，再插补。连续列用中位数；二元高峰标志使用众数，避免 0.5；充电桩类别使用众数。全列无有效值会阻止处理。
4. 缺失目标类别不猜测：可保留在分析数据中，但训练时排除。未知目标拼写报错。
5. 原文背景的 `Grid Heavy` 统一映射到任务 2 的 `Grid_Surge`，并记录映射条数。应向老师确认此映射。
6. 功率：`Power_kW = Energy_kWh / (Duration_Mins / 60)`。分组约定为 ≤22 kW、22–50 kW、>50 kW。原文只要求“能源效率分类”而未给定义，因此这里使用功率分组作为分析代理；**它不是电能转换效率**。
7. 原文字典没有收入和电价。页面优先使用可用的 `Revenue` 数值字段；否则显示“估算高峰收入”，由可调假设电价计算。默认 0.50 RM/kWh 是演示假设，不是学校实际电价。

## 机器学习七步流程

1. **定义问题**：基于充电桩、能量、时长、时段、温度，预测三种电网状态。
2. **收集数据**：读取原始 CSV 并核对字段；正式结果必须来自老师数据。
3. **准备数据**：先去重与校验，再分层划分 80% 训练集和 20% 测试集。插补、标准化和编码放入 sklearn Pipeline，仅在训练数据上拟合。
4. **选择模型**：随机森林、决策树、KNN；采用固定且明确的超参数。
5. **训练与比较**：仅在训练集内进行最多五折分层交叉验证，以宏平均 F1 比较模型。固定随机种子 42。
6. **独立评估**：对 20% 留出集报告准确率、宏平均精确率/召回率/F1、分类别指标和混淆矩阵。模型选择依据是训练集交叉验证分数，不用测试集来挑模型。
7. **保存与使用**：保存完整预处理与分类器及元数据到 `ev_model.pkl`，网页以同一 Pipeline 预测。保存的是在 80% 训练集上拟合的最佳模型。

EDA 的清洗文件采用全数据中位数用于描述分析；**训练直接读取原始 CSV**，不使用已经全局插补的 EDA 文件，避免测试集信息泄漏。派生功率与原始能量/时长存在确定关系，本实现选择五个基础输入特征训练，保留功率用于分析与展示。

## 输出文件

```text
cleaned_ev_dataset.csv
ev_model.pkl
reports/
  data_quality.json              # 缺失、无效值、去重、插补值、文件指纹
  group_summary.csv             # 充电桩 × 高峰标志分组统计
  model_comparison.csv          # CV 分数、独立测试集指标
  classification_reports.json  # 每个类别的 precision/recall/F1/support
  training_metadata.json       # 数据来源、划分、模型版本等
  confusion_RandomForest.png
  confusion_DecisionTree.png
  confusion_KNN.png
```

演示模式把同样的产物放在 `demo/`，同时在页面和模型元数据中标识 `demo=true`。

## 参考文档

- [Streamlit 官方测试文档](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Streamlit Community Cloud 部署](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit 部署依赖](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

## 作业中的 AI 使用说明

本项目使用了 AI 辅助生成代码、中文注释和测试。原作业声明对 AI 使用有具体要求，请理解并检查实现，按教师允许的范围使用，并如实保留提示词、生成内容及自己的修改记录。此项目不包含代签的学生声明或虚构的实验结果。
