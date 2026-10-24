"""一键运行任务 1 和任务 2；默认读取项目根目录中的老师原始数据。"""
import argparse
from pathlib import Path

from data_pipeline import build_dataset
from ev_core import ROOT
from train_model import train


def run(source: Path, output: Path):
    build_dataset(source, output)
    metadata = train(source, output)
    print(f"正式流程完成，模型：{metadata['selected_model']}，输出：{output}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ev_campus_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        run(args.data, args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"无法完成流程：{exc}\n请检查原始 CSV 文件是否已放入项目目录。\n")
