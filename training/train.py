# -*- coding: utf-8 -*-
"""
============================================================
 YOLO11 电动车进电梯检测 —— 训练脚本
============================================================
用法: 在 PyCharm 中打开 E:\\mytrain 项目(解释器选 yolo 环境),
     右键本文件 → Run 'train'

训练完成后产出:
  - 最优模型: E:/mytrain/runs/yolo11n_elevator/weights/best.pt
  - 训练曲线: E:/mytrain/runs/yolo11n_elevator/results.png
  - 混淆矩阵: E:/mytrain/runs/yolo11n_elevator/confusion_matrix.png
预计耗时: RTX 5060 显卡约 3.5 ~ 4 小时(8000张子集, batch=32+8进程,已实测验证)
============================================================
"""
from ultralytics import YOLO

# ==================== 可调参数区 ====================
MODEL   = "E:/mytrain/yolo11n.pt"   # 预训练权重(已提前下载好,无需联网)
DATA    = "E:/mytrain/data_sub8000.yaml"   # 8000张抽样子集(Windows最稳方案,数据由prepare_subset.py生成)
EPOCHS  = 60          # 训练轮数(智能体已完成全链路试跑验证,直接正式跑即可)
IMGSZ   = 640         # 图片尺寸,数据集就是640,不要改
BATCH   = 32          # 一次喂给显卡的图片数。实测32比16快约3倍(RTX 5060 8GB够用);
                      # 若报 CUDA out of memory 就改回 16
DEVICE  = 0           # 0 = 用NVIDIA显卡(推荐); "cpu" = 用CPU(慢几十倍,不推荐)
WORKERS = 8           # 数据加载进程数(已实测8进程+batch32最合适;脚本有__main__保护不会套娃;
                      # 万一哪天卡死不动,把此值改 0 —— 永不卡死但慢约10倍)
PROJECT = "E:/mytrain/runs"
NAME    = "yolo11n_elevator"
# ====================================================


def main():
    print("=" * 60)
    print("开始训练 YOLO11 电动车进电梯检测模型")
    print(f"  数据集: {DATA}")
    print(f"  预训练权重: {MODEL}")
    print(f"  轮数={EPOCHS}, 批大小={BATCH}, 设备={'GPU(显卡)' if DEVICE == 0 else DEVICE}")
    print("=" * 60)

    # 加载预训练权重(在 COCO 数据集上预训练过的 yolo11n)
    model = YOLO(MODEL)

    results = model.train(
        data=DATA,          # 数据集配置
        epochs=EPOCHS,      # 训练轮数
        imgsz=IMGSZ,        # 输入图片尺寸
        batch=BATCH,        # 批大小
        device=DEVICE,      # 训练设备
        workers=WORKERS,    # 数据加载进程数
        project=PROJECT,    # 结果保存目录
        name=NAME,          # 本次训练的文件夹名
        patience=25,        # 连续25轮验证精度不再提升 → 自动提前停止,不浪费时间
        seed=42,            # 固定随机种子,保证可复现
        exist_ok=True,      # 重复运行时沿用同一个结果文件夹
    )

    print()
    print("=" * 60)
    print("训练完成!所有产出都在: E:/mytrain/runs/" + NAME)
    print("  - 最优模型(拿去用): runs/" + NAME + "/weights/best.pt")
    print("  - 最后一轮模型:     runs/" + NAME + "/weights/last.pt")
    print("  - 训练曲线图:       runs/" + NAME + "/results.png")
    print("=" * 60)
    print("下一步: ① 右键运行 val.py 拿评估指标  ② 右键运行 predict_test.py 看检测效果")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("\n[报错] 显存不够。请关闭游戏/浏览器等占显存的程序后重试,")
        print("       或把 train.py 顶部的 BATCH = 16 改成 BATCH = 8")
    except ModuleNotFoundError as e:
        print(f"\n[报错] 缺少依赖库: {e}")
        print("       请先在 PyCharm 终端执行(第0步的安装命令):")
        print(r'       D:\anaconda3\envs\yolo\python.exe -m pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple')
    except FileNotFoundError as e:
        print(f"\n[报错] 找不到文件: {e}")
        print("       请检查 E:/mytrain/data.yaml 里的数据集路径是否正确")
    except Exception as e:
        msg = str(e)
        print(f"\n[报错] {type(e).__name__}: {msg[:300]}")
        if "out of memory" in msg.lower():
            print("       → 显存不足: 关掉占显存的程序,或把 BATCH 改成 8")
        elif "No such file" in msg:
            print("       → 文件路径问题: 检查数据集文件夹是否被移动/重命名")
        else:
            print("       → 把上面这段报错完整发给智能体,帮你分析")
