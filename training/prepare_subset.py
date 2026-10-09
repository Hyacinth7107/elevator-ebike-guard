# -*- coding: utf-8 -*-
"""
============================================================
 生成训练子集 —— 从 28000 张全量数据集随机抽样 8000 张到 E:/mytrain/dataset_sub8000
============================================================
为什么抽样:
  Windows 下数据加载进程数 workers=0 最稳定(避免多进程假死卡住),
  但单线程加载 26414 张全量数据每轮太慢;抽 8000 张后每轮约 2 分钟,
  60 轮约 2 小时,质量足够(电梯场景2类检测,8000张已很充足)。
  验证集保持全量 1276 张(评估指标更可信)。

用法: 右键 Run 'prepare_subset' (整个项目生命周期只需运行一次)
============================================================
"""
import os
import random
import shutil

SRC = r"E:\电动车检测数据集\04_主力训练集_28000张\电梯内自行车与电动车目标检测数据集-解压后可直接使用"
DST = r"E:\mytrain\dataset_sub8000"
N = 8000  # 抽样数量

random.seed(42)  # 固定随机种子,每次抽样结果一致

src_img = os.path.join(SRC, "train", "images")
src_lbl = os.path.join(SRC, "train", "labels")
dst_img = os.path.join(DST, "train", "images")
dst_lbl = os.path.join(DST, "train", "labels")
dst_val_img = os.path.join(DST, "valid", "images")
dst_val_lbl = os.path.join(DST, "valid", "labels")

for d in (dst_img, dst_lbl, dst_val_img, dst_val_lbl):
    os.makedirs(d, exist_ok=True)

# ---- 1. 抽样复制 train ----
names = [f for f in os.listdir(src_img) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"源train图片总数: {len(names)}, 随机抽取 {N} 张...")
picked = random.sample(names, N)

for i, f in enumerate(picked, 1):
    stem = os.path.splitext(f)[0]
    shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
    lp = os.path.join(src_lbl, stem + ".txt")
    if os.path.exists(lp):
        shutil.copy2(lp, os.path.join(dst_lbl, stem + ".txt"))
    if i % 1000 == 0:
        print(f"  已复制 {i}/{N}")
print(f"train 子集复制完成: {N} 对 图片+标签")

# ---- 2. 全量复制 valid ----
sv = os.listdir(os.path.join(SRC, "valid", "images"))
for f in sv:
    stem = os.path.splitext(f)[0]
    shutil.copy2(os.path.join(SRC, "valid", "images", f), os.path.join(dst_val_img, f))
    lp = os.path.join(SRC, "valid", "labels", stem + ".txt")
    if os.path.exists(lp):
        shutil.copy2(lp, os.path.join(dst_val_lbl, stem + ".txt"))
print(f"valid 全量复制完成: {len(sv)} 张")

# ---- 3. 自检 ----
n_img = len(os.listdir(dst_img))
n_lbl = len(os.listdir(dst_lbl))
print()
print(f"子集自检: train图片 {n_img} / train标签 {n_lbl} / valid图片 {len(os.listdir(dst_val_img))}")
if n_img == n_lbl == N:
    print("DONE! 子集生成成功,位置:", DST)
else:
    print("警告: 图片标签数量不匹配,请把上面信息发给智能体")
