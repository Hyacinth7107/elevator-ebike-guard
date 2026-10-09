# -*- coding: utf-8 -*-
"""
============================================================
 模型评估脚本 —— 算出 mAP 等指标(技术报告必须写这些数字)
============================================================
用法: 训练完成后,在 PyCharm 中右键 Run 'val'
说明: 用"全量数据集"自带的测试集(622张)评估,衡量真实水平,
     与训练用的8000张子集互不干扰。
产出: 控制台打印 mAP50 / mAP50-95 / 各类别精度
============================================================
"""
from ultralytics import YOLO
import glob

DATA = "E:/mytrain/data.yaml"   # 全量数据集(用它的 test 测试集评估)


def main():
    # 自动寻找最新一次训练的最优权重
    candidates = sorted(glob.glob("E:/mytrain/runs/*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("没找到训练结果! 请先运行 train.py 完成训练")

    best = candidates[-1]
    print("使用权重:", best)

    model = YOLO(best)
    # 在"测试集"(622张从未参与训练的图)上评估,图表存到 E 盘
    metrics = model.val(
        data=DATA, split="test", device=0, workers=4,
        project="E:/mytrain/runs", name="val_on_test", exist_ok=True,
    )

    # 每类 mAP50: box.ap 是 (类别数, 10) 数组,第0列即 IoU=0.50 的 AP
    try:
        per_map50 = list(metrics.box.ap[:, 0])
    except Exception:
        per_map50 = [None] * len(metrics.box.maps)

    print()
    print("=" * 62)
    print("评估结果(抄进技术报告):")
    print("=" * 62)
    print(f"  总体 mAP50     = {metrics.box.map50:.4f}")
    print(f"  总体 mAP50-95  = {metrics.box.map:.4f}")
    print()
    print("  各类别指标:")
    for i in range(len(metrics.box.maps)):
        name = ["自行车 bicycle", "电动车 ebike"][i]
        m50 = f"{per_map50[i]:.4f}" if per_map50[i] is not None else "见上方表格"
        print(f"    {name}:  mAP50={m50}  |  mAP50-95={metrics.box.maps[i]:.4f}")
    print()
    print("  速度(每张图): 预处理{:.1f}ms 推理{:.1f}ms".format(
        metrics.speed["preprocess"], metrics.speed["inference"]))
    print("=" * 62)
    print("评估图表(PR曲线/混淆矩阵等)已保存到: E:/mytrain/runs/val_on_test/")


if __name__ == "__main__":   # Windows 下必须加保护
    main()
