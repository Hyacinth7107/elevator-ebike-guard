# -*- coding: utf-8 -*-
"""
============================================================
 推理测试脚本 —— 用训练好的模型实际检测图片,看效果
============================================================
用法: 训练完成后,在 PyCharm 中右键 Run 'predict_test'
产出: 带检测框的结果图,保存在 E:/mytrain/预测结果/
     - batch_0: 别人项目里的电梯实拍测试图(看"陌生环境"表现)
     - batch_1: 数据集自带的测试集622张(看"考试内"表现)
============================================================
"""
from ultralytics import YOLO
import glob
import os


def main():
    # 1. 自动寻找最新训练出的最优权重
    candidates = sorted(glob.glob("E:/mytrain/runs/*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("没找到训练结果! 请先运行 train.py 完成训练")
    best = candidates[-1]
    print("使用权重:", best)
    model = YOLO(best)

    # 2. 要检测的图片文件夹
    folders = [
        # 别人电梯项目里的41张实拍测试图(检验模型泛化能力)
        r"E:\电动车检测数据集\01_Lily-Le电梯电动车检测项目\ebikedetection-main\test_samples",
        # 数据集测试集(检验常规表现)
        r"E:\电动车检测数据集\04_主力训练集_28000张\电梯内自行车与电动车目标检测数据集-解压后可直接使用\test\images",
    ]

    save_root = r"E:\mytrain\预测结果"
    os.makedirs(save_root, exist_ok=True)

    for i, folder in enumerate(folders):
        if not os.path.isdir(folder):
            print("[跳过] 文件夹不存在:", folder)
            continue
        print(f"\n正在检测第 {i + 1} 组图片: {folder}")
        results = model.predict(
            source=folder,
            conf=0.35,            # 置信度阈值:只有把握>35%才算检出(可调0.25~0.5)
            save=True,
            project=save_root,
            name=f"batch_{i}",
            exist_ok=True,
            max_det=100,
            device=0,             # 用显卡推理
        )
        total_boxes = sum(len(r.boxes) for r in results)
        n_detected_img = sum(1 for r in results if len(r.boxes) > 0)
        print(f"  处理 {len(results)} 张图, 检出 {total_boxes} 个目标, {n_detected_img} 张图有目标")

    print()
    print("=" * 60)
    print("完成! 带检测框的结果图保存在: E:/mytrain/预测结果/")
    print("打开看看: 框是否贴合车辆、类别对不对、有没有漏检/误检")
    print("=" * 60)


if __name__ == "__main__":   # Windows 下必须加保护
    main()
