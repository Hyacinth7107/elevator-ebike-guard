# -*- coding: utf-8 -*-
"""
============================================================
 GradCAM v2 —— 类别定向热力图(梯度对准 ebike 类)
============================================================
v1 的 EigenCAM 是全局主成分,被高对比文字吸引;
v2 用 GradCAM + 自定义 target(取检测头 ebike 通道的最大响应),
梯度反传后热区只跟随"电动车"类别相关区域。
产出: E:/mysystem/材料/技术加分/gradcam_热力图.png (覆盖 v1)
============================================================
"""
import glob
import os

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

WEIGHTS = r"E:\mytrain\runs\yolo11n_elevator_100final\weights\best.pt"
OUT = r"E:\mysystem\材料\技术加分"
TEST_IMG = (r"E:\电动车检测数据集\04_主力训练集_28000张"
            r"\电梯内自行车与电动车目标检测数据集-解压后可直接使用\test\images")
N_SHOW = 3
SIZE = 640
NC = 2  # bicycle, ebike —— ebike 类别索引=1

model = YOLO(WEIGHTS)
print("模型加载完成")

# 选 3 张有高置信电动车的图
cands = sorted(glob.glob(os.path.join(TEST_IMG, "*.jpg")))[:80]
picked = []
for f in cands:
    r = model.predict(f, conf=0.60, device=0, imgsz=640, verbose=False)[0]
    eb = [b for b in r.boxes if model.names[int(b.cls[0])] == "ebike"]
    if eb:
        picked.append((f, max(float(b.conf[0]) for b in eb)))
    if len(picked) >= N_SHOW:
        break
print(f"选中 {len(picked)} 张")

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.model.to(torch_device)
net.eval()


class EbikeTarget:
    """从 YOLO 检测头输出 (1, 4*(4+NC), 8400) 取 ebike 类分数的最大 anchor 响应"""

    def __call__(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 2:      # 某些路径会剥掉 batch 维 -> 补回
            output = output.unsqueeze(0)
        # 每 (4+NC)=6 个通道一组, ebike = 组内第 6 通道(索引5,11,17...)
        n_ch = output.shape[1]
        ebike_chs = list(range(4 + 1, n_ch, 4 + NC))
        scores = [output[0, c, :].max() for c in ebike_chs]
        return torch.stack(scores).max()


class WrapNet(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.net = m

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


wrap = WrapNet(net).to(torch_device).eval()
target_layers = [net.model[-2]]
cam = GradCAM(wrap, target_layers)
print("GradCAM(类别定向) 就绪")

panels = []
for f, conf in picked:
    bgr = cv2.imread(f)
    rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    disp = cv2.resize(rgb0, (SIZE, SIZE)) / 255.0

    x = cv2.resize(bgr, (SIZE, SIZE)).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    tensor = torch.from_numpy(x).to(torch_device).requires_grad_(True)  # CAM 反传需要

    grayscale = cam(tensor, targets=[EbikeTarget()])[0]
    if grayscale.ndim == 3:
        grayscale = grayscale[0]
    g = grayscale - grayscale.min()
    if g.max() > 0:
        g = g / g.max()
    g = cv2.resize(g, (SIZE, SIZE))

    overlay = show_cam_on_image(disp, g, use_rgb=True)
    panels.append((disp, overlay, conf))
    print(f"  完成 1 张 (置信度{conf:.0%})")

if panels:
    fig, axes = plt.subplots(len(panels), 2, figsize=(7.2, 3.4 * len(panels)))
    if len(panels) == 1:
        axes = axes[None]
    for i, (orig, over, conf) in enumerate(panels):
        axes[i][0].imshow(orig)
        axes[i][0].set_title(f"原图 {i+1}", fontsize=11)
        axes[i][1].imshow(over)
        axes[i][1].set_title(f"GradCAM 热力图·定向电动车类 (置信度{conf:.0%})", fontsize=11)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("GradCAM 决策可视化: 热区跟随电动车主体 —— 模型依据目标本身特征而非背景做判断",
                 fontsize=12.5, y=0.995)
    fig.tight_layout()
    out_png = os.path.join(OUT, "gradcam_热力图.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out_png)
