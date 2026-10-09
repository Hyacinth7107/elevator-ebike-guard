# -*- coding: utf-8 -*-
"""
============================================================
 恶劣条件鲁棒性扩展 —— "鲁棒性测试+4分"的加固证据
============================================================
 设计: 取60张真值含电动车的电梯测试图, 每张生成3种劣化版本:
   轻度模糊(高斯模糊7×7, 模拟镜头脏污/失焦)
   暗光(亮度×0.45, 模拟夜间/背光)
   高斯噪声(σ=25, 模拟低质摄像头/信号干扰)
 同一模型一次推理(conf=0.10), 三档阈值内存过滤(与主鲁棒性测试同口径)。
 产出: 材料\\技术加分\\恶劣条件鲁棒性.png + 恶劣条件数据.txt
       鲁棒性结果\\恶劣条件效果图\\(同一辆车四种条件对比样例)
 用法: D:\\anaconda3\\envs\\yolo\\python.exe E:\\mysystem\\材料\\技术加分\\gen_adverse.py
============================================================
"""
import glob
import os
import time

import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

WEIGHTS = r"E:\mytrain\runs\yolo11n_elevator_100final\weights\best.pt"
OUT = r"E:\mysystem\材料\技术加分"
EFF_DIR = r"E:\mysystem\鲁棒性结果\恶劣条件效果图"
TEST_IMG = (r"E:\电动车检测数据集\04_主力训练集_28000张"
            r"\电梯内自行车与电动车目标检测数据集-解压后可直接使用\test\images")
TEST_LBL = TEST_IMG.replace("images", "labels")
THRESHOLDS = [0.25, 0.35, 0.50]
N_IMG = 60

CONDS = [
    ("orig",  "原始图像"),
    ("blur",  "轻度模糊"),
    ("dark",  "暗光×0.45"),
    ("noise", "噪声σ=25"),
]


def perturb(img, kind, rng):
    if kind == "blur":
        return cv2.GaussianBlur(img, (7, 7), 0)
    if kind == "dark":
        return np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if kind == "noise":
        noise = rng.normal(0, 25, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def pick_positive(n):
    """真值含电动车(类别1)的测试图, 与主鲁棒性测试同规则"""
    out = []
    for f in sorted(glob.glob(os.path.join(TEST_IMG, "*.jpg"))):
        lbl = os.path.join(TEST_LBL, os.path.splitext(os.path.basename(f))[0] + ".txt")
        if os.path.exists(lbl):
            with open(lbl, encoding="utf-8") as fh:
                if any(ln.strip().startswith("1") for ln in fh):
                    out.append(f)
        if len(out) >= n:
            break
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(EFF_DIR, exist_ok=True)
    files = pick_positive(N_IMG)
    print(f"样本: {len(files)} 张真值含电动车的测试图")

    model = YOLO(WEIGHTS)
    names = model.names
    rng = np.random.default_rng(42)
    stat = {k: {t: {"hit": 0, "n": 0, "conf_sum": 0.0} for t in THRESHOLDS}
            for k, _ in CONDS}
    best_demo = None  # (原始置信度, results列表)

    t0 = time.time()
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        arrs = [perturb(img, k, rng) for k, _ in CONDS]
        results = model.predict(arrs, conf=0.10, device=0, imgsz=640,
                                stream=False, verbose=False)
        per_cond_conf = {}
        for (key, _), r in zip(CONDS, results):
            eb = [float(b.conf[0]) for b in r.boxes
                  if names[int(b.cls[0])] == "ebike"]
            cmax = max(eb) if eb else 0.0
            per_cond_conf[key] = cmax
            for th in THRESHOLDS:
                s = stat[key][th]
                s["n"] += 1
                if cmax >= th:
                    s["hit"] += 1
                    s["conf_sum"] += cmax
        if best_demo is None or per_cond_conf["orig"] > best_demo[0]:
            best_demo = (per_cond_conf["orig"], results)
    used = time.time() - t0
    print(f"推理完成, 耗时 {used:.1f}s")

    # ---- 保存同一辆车四种条件的检测效果图(PPT用) ----
    if best_demo:
        for (key, cname), r in zip(CONDS, best_demo[1]):
            try:
                r.save(filename=os.path.join(
                    EFF_DIR, f"样例_{cname}.jpg"))
            except Exception as e:
                print("效果图保存失败:", e)

    # ---- 汇总数字 ----
    def rate(k, t):
        s = stat[k][t]
        return s["hit"] / s["n"] * 100 if s["n"] else 0.0

    def avgc(k, t=0.35):
        s = stat[k][t]
        return s["conf_sum"] / s["hit"] * 100 if s["hit"] else 0.0

    n_real = stat["orig"][0.35]["n"]
    txt_lines = [
        f"恶劣条件鲁棒性扩展 (n={n_real}张真值含电动车测试图, 同一模型, 三档阈值, 与主鲁棒性测试同口径)",
        f"条件           @25%    @35%(默认)  @50%    平均最高置信度@35%",
    ]
    for k, cname in CONDS:
        txt_lines.append(
            f"{cname:<12}  {rate(k,0.25):5.1f}%   {rate(k,0.35):5.1f}%   "
            f"{rate(k,0.50):5.1f}%   {avgc(k):5.1f}%")
    o35, d35 = rate("orig", 0.35), rate("dark", 0.35)
    b35, n35r = rate("blur", 0.35), rate("noise", 0.35)
    txt_lines.append(
        f"结论: 默认阈值0.35下, 轻度模糊/暗光/噪声使检出率由{o35:.1f}%变为 "
        f"{b35:.1f}%/{d35:.1f}%/{n35r:.1f}%, 量化了系统对画质劣化的耐受边界。")
    txt_lines.append(
        "说明: 模糊=高斯模糊7×7(镜头脏污/失焦), 暗光=亮度×0.45(夜间/背光), "
        "噪声=高斯σ=25(低质摄像头)。")

    # ---- 图表: 4条件 × 3阈值 分组柱状图 ----
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(CONDS))
    width = 0.26
    colors = ["#93c5fd", "#3b82f6", "#1d4ed8"]
    for i, th in enumerate(THRESHOLDS):
        vals = [rate(k, th) for k, _ in CONDS]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=f"阈值 {th:.0%}", color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_ylim(0, 112)
    ax.set_ylabel("电动车检出率 (%)")
    ax.set_title(f"恶劣条件下的检出率 (n={n_real}, 真值均含电动车)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "恶劣条件鲁棒性.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "恶劣条件数据.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print()
    for ln in txt_lines:
        print(" ", ln)
    print()
    print("产出:", os.path.join(OUT, "恶劣条件鲁棒性.png"))
    print("     ", os.path.join(OUT, "恶劣条件数据.txt"))
    print("效果图:", EFF_DIR)


if __name__ == "__main__":
    main()
