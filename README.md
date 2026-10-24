# EV Campus：校园充电数据分析与机器学习

Assessment 2（ITS72604）对应的 Python 项目。代码、关键设计说明和操作指引均带中文注释。

## 当前交付状态

- 已实现数据清洗、特征工程、分组分析、三种分类模型、评估图表和三模块 Streamlit 门户。
- 已配置独立 `.venv`，并提供数据处理训练、网页启动两个 PyCharm 运行配置。
- 已接入 2026-10-24 提供的课程 CSV（1,200 条会话），按实际字段完成清洗和正式训练。
- 电量缺失 35 条、时长缺失 25 条，分别用中位数 13.28 kWh、97 分钟填补；没有重复会话或缺失标签。训练插补只在训练集内拟合。
- 正式的 `cleaned_ev_dataset.csv`、`ev_model.pkl` 和 `reports/` 已生成在项目根目录。原始 CSV 保存在本机，默认不提交。
- GitHub 公开仓库：[hezishen846-eng/ev-campus-assessment2](https://github.com/hezishen846-eng/ev-campus-assessment2)。
- 在线网站：[EV Campus 充电运营分析](https://ev-campus-hezishen846.streamlit.app/)（Streamlit Community Cloud）。
- 演示视频：由作者另行录制上传，完成后补充公开链接。
- 页面只使用课程数据与正式模型；已移除演示目录、生成脚本及演示模式。

## 本次训练结果

固定随机种子 42，分层划分 960 条训练记录、240 条测试记录。仅根据训练集五折交叉验证的宏平均 F1 选择模型，最终保存 **DecisionTree**。

| 模型 | 训练 CV 宏 F1 | 测试准确率 | 测试宏精确率 | 测试宏召回率 | 测试宏 F1 |
|---|---:|---:|---:|---:|---:|
| DecisionTree | 0.8169 | 80.83% | 0.8022 | 0.8989 | 0.8124 |
| RandomForest | 0.8074 | 81.67% | 0.8031 | 0.8947 | 0.8161 |
| KNN | 0.6376 | 78.75% | 0.7182 | 0.6319 | 0.6660 |

随机森林本次测试准确率略高，但测试集不参与选模型，避免按测试结果挑选造成评估偏差。以上是本次课程数据的实验结果，不代表已验证的校园生产系统性能。

## 在 PyCharm 中运行

项目已打开，并使用本项目 `.venv/bin/python`（Python 3.12）作为解释器。

右上角有两个运行配置：

1. **01 正式数据处理与训练**：先把老师的 CSV 放到项目根目录，命名为 `ev_campus_dataset.csv`，再运行。没有文件时会明确提示缺少数据。
2. **02 启动充电分析网页**：启动 Streamlit，按运行窗口中的链接打开浏览器。通常为 `http://127.0.0.1:8501`。

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

# 启动网页
python -m streamlit run app.py --server.address 127.0.0.1

# 功能测试（网页测试使用项目根目录的正式产物）
python -m pytest -q
```

## 文件与评分要求的对应关系

| 任务 | 文件 | 完成内容 |
|---|---|---|
| 任务 1：6 分 | `data_pipeline.py`、`ev_core.py` | 校验、缺失值处理、去重、功率、分组 CSV、清洗审计 |
| 任务 2：9 分 | `train_model.py` | 80/20 分层划分、独热编码、随机森林/决策树/KNN 对比、四项指标、混淆矩阵、模型持久化 |
| 任务 3：9 分 | `app.py` | KPI 仪表盘、交互图表、实时预测，侧边栏导航 |
| 任务 4：6 分 | `.streamlit/`、`requirements.txt`、`docs/部署与演示.md` | 统一界面、依赖配置、GitHub 与 Streamlit Cloud 已发布；视频待作者录制 |
| 验证 | `tests/` | 数据异常、去重、无泄漏插补、模型重新加载、筛选和实时预测 |

## 数据处理约定

1. 使用 CSV 的实际九列：`Session_ID`、`Charger_Type`、`User_Type`、`Start_Hour`、`Duration_Mins`、`Energy_kWh`、`Overstay_Mins`、`Battery_Temp_C`、`Grid_Status_Label`。充电桩为 AC/DC，用户为 Student/Staff/Visitor；保留原始含义，不把电池温度改名为环境温度。
2. 无会话编号的记录删除；重复编号保留第一条；删除数量记录在审计报告中。
3. 非数值、无穷值及语义无效的输入转为缺失，再按中位数插补；类别用众数。开始小时应为 0–23 的整数，电量、时长应大于 0，超时应非负，温度应高于绝对零度。不会用旧字典的 5 kWh 下限或 24–42°C 环境温度范围误删当前数据。未知类别报错，全列无有效值会阻止处理。
4. 缺失目标类别不猜测：可保留在分析数据中，但训练时排除。未知目标拼写报错。
5. 当前 CSV 已使用 Normal、Overheated、Grid_Surge，无需目标改名。保留旧别名 Grid Heavy 的兼容映射和审计记录。
6. 功率：`Power_kW = Energy_kWh / (Duration_Mins / 60)`。分组约定为 ≤22 kW、22–50 kW、>50 kW。原文只要求“能源效率分类”而未给定义，因此这里使用功率分组作为分析代理；**它不是电能转换效率**。
7. CSV 没有高峰标志。依据原作业的 12:00–17:00，派生 `Peak_Hour_Flag = 1` 当 `12 <= Start_Hour < 17`。这只标记开始小时，不拆分跨时段会话。
8. CSV 没有收入和电价，因此页面显示“估算高峰收入”，由高峰开始会话的电量乘可调假设电价得到。默认 0.50 RM/kWh 是估算假设，不是学校实际电价或账单。若有 Revenue 列则优先使用。

## 机器学习七步流程

1. **定义问题**：基于充电桩、用户类型、开始小时、电量、时长、超时、电池温度，分类三种电网状态。使用会话统计量，不能声称为开始充电前的预测。
2. **收集数据**：读取原始 CSV 并核对字段；正式结果必须来自老师数据。
3. **准备数据**：先去重与校验，再分层划分 80% 训练集和 20% 测试集。插补、标准化和编码放入 sklearn Pipeline，仅在训练数据上拟合。
4. **选择模型**：随机森林、决策树、KNN；采用固定且明确的超参数。
5. **训练与比较**：仅在训练集内进行最多五折分层交叉验证，以宏平均 F1 比较模型。固定随机种子 42。
6. **独立评估**：对 20% 留出集报告准确率、宏平均精确率/召回率/F1、分类别指标和混淆矩阵。模型选择依据是训练集交叉验证分数，不用测试集来挑模型。
7. **保存与使用**：保存完整预处理与分类器及元数据到 `ev_model.pkl`，网页以同一 Pipeline 预测。保存的是在 80% 训练集上拟合的最佳模型。

EDA 的清洗文件采用全数据中位数用于描述分析；**训练直接读取原始 CSV**，不使用已经全局插补的 EDA 文件，避免测试集信息泄漏。使用七个实际基础字段训练，其中 Charger_Type 和 User_Type 独热编码；Session_ID 和目标标签不作为输入。派生功率、高峰标志用于统计分析，不重复作为模型输入。

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

正式产物的元数据标记 `demo=false`、`schema_version=campus_sessions_v2`，并记录原始文件、清洗文件的 SHA-256 指纹。网页核对模型与数据版本，避免旧模型混用。`tests/conftest.py` 的小样本仅供边界测试，不会用于网页或正式模型。

## 参考文档

- [Streamlit 官方测试文档](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Streamlit Community Cloud 部署](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit 部署依赖](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

## 作业中的 AI 使用说明

本项目使用了 AI 辅助生成代码、中文注释和测试。原作业声明对 AI 使用有具体要求，请理解并检查实现，按教师允许的范围使用，并如实保留提示词、生成内容及自己的修改记录。此项目不包含代签的学生声明或虚构的实验结果。
