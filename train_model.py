"""任务 2：分层划分、训练集内交叉验证选模型、独立测试集评估与模型保存。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # 无桌面服务器也能生成图表文件。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ev_core import CATEGORICAL, FEATURES, LABELS, NUMERIC, ROOT, SCHEMA_VERSION, TARGET, file_hash, normalize_data


def make_pipeline(estimator) -> Pipeline:
    # 插补器、标准化和独热编码都在 Pipeline 内拟合，因此每个交叉验证折相互隔离。
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categories", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([("prepare", preprocessor), ("classifier", estimator)])


def train(source: Path, output: Path) -> dict:
    # 学习用输入来自原始 CSV，而非已经使用全数据中位数插补的 EDA 文件。
    df, audit = normalize_data(pd.read_csv(source))
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    counts = df[TARGET].value_counts()
    if set(counts.index) != set(LABELS) or counts.min() < 5:
        raise ValueError("训练需要 Normal、Overheated、Grid_Surge 三类，每类至少 5 条有效记录。")
    if len(df) < 30:
        raise ValueError("有效训练记录少于 30 条，无法进行可靠的分层划分与交叉验证。")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET],
    )
    for col in FEATURES:
        if X_train[col].isna().all():
            raise ValueError(f"训练集的 {col} 整列为空，无法学习预处理规则。")
    folds = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_leaf=2,
                                                class_weight="balanced", random_state=42, n_jobs=1),
        "DecisionTree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    results, fitted, details = [], {}, {}
    for name, estimator in candidates.items():
        model = make_pipeline(estimator)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", error_score="raise")
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        report = classification_report(y_test, prediction, labels=LABELS, output_dict=True, zero_division=0)
        results.append({"model": name, "cv_f1_macro": float(scores.mean()), "cv_std": float(scores.std()),
                        "accuracy": accuracy_score(y_test, prediction), "precision_macro": report["macro avg"]["precision"],
                        "recall_macro": report["macro avg"]["recall"], "f1_macro": report["macro avg"]["f1-score"]})
        details[name] = report
        fitted[name] = model
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(confusion_matrix(y_test, prediction, labels=LABELS), annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set(xlabel="Predicted label", ylabel="True label", title=f"{name} | Test confusion matrix")
        fig.tight_layout()
        fig.savefig(reports / f"confusion_{name}.png", dpi=160)
        plt.close(fig)
    comparison = pd.DataFrame(results).sort_values("cv_f1_macro", ascending=False, kind="stable")
    best = str(comparison.iloc[0]["model"])
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    (reports / "classification_reports.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    metadata = {"demo": False, "schema_version": SCHEMA_VERSION,
                "source_file": source.name, "selected_model": best, "selection_rule": "training CV macro F1",
                "train_rows": len(X_train), "test_rows": len(X_test), "cv_folds": folds, "random_state": 42,
                "source_sha256": file_hash(source), "sklearn_version": sklearn.__version__,
                "missing_targets_excluded": audit["missing_targets"], "features": FEATURES,
                "class_counts": {k: int(v) for k, v in counts.items()},
                "train_session_ids": df.loc[X_train.index, "Session_ID"].tolist(),
                "test_session_ids": df.loc[X_test.index, "Session_ID"].tolist()}
    cleaned = output / "cleaned_ev_dataset.csv"
    if cleaned.exists():
        metadata["cleaned_sha256"] = file_hash(cleaned)
    # 模型中包含完整的预处理 Pipeline；网页端不要另写一套编码规则。
    joblib.dump({"pipeline": fitted[best], "metadata": metadata}, output / "ev_model.pkl")
    (reports / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ev_campus_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = train(args.data, args.output_dir)
        print(f"模型训练完成，选用：{result['selected_model']}")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"训练失败：{exc}\n")
