# -*- coding: utf-8 -*-
"""
============================================================
 鲁棒性测试脚本 —— 评分表"鲁棒性测试 +4分"的证据生成器
============================================================
测试设计(三组 × 三档阈值):
  A组 正样本: 电梯监控图且真值含电动车(200张) → 检出率/平均置信度
  B组 负样本: 电梯监控图但真值无电动车(39张)   → 电动车误报率
  C组 跨域:   陌生电梯实拍图(12张,无真值)       → 观察泛化(保存效果图)
  阈值档位: 0.25(宽松) / 0.35(默认) / 0.50(严格) —— 同一次推理内存过滤

用法: D:\\anaconda3\\envs\\yolo\\python.exe E:\\mysystem\\test_robustness.py
产出: E:\\mysystem\\鲁棒性结果\\robustness_detail.csv + 汇总报告.md + C组效果图
============================================================
"""
import csv
import glob
import os
import time

from ultralytics import YOLO

DATA_ROOT = (r"E:\电动车检测数据集\04_主力训练集_28000张"
             r"\电梯内自行车与电动车目标检测数据集-解压后可直接使用")
TEST_IMG = os.path.join(DATA_ROOT, "test", "images")
TEST_LBL = os.path.join(DATA_ROOT, "test", "labels")
CROSS_DIR = r"E:\电动车检测数据集\01_Lily-Le电梯电动车检测项目\ebikedetection-main\test_samples"
OUT_DIR = r"E:\mysystem\鲁棒性结果"
RUNS_DIR = r"E:\mytrain\runs"
THRESHOLDS = [0.25, 0.35, 0.50]
N_POS = 200  # A组抽样数


def find_latest_best():
    cands = glob.glob(os.path.join(RUNS_DIR, "*", "weights", "best.pt"))
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    weights = find_latest_best()
    if not weights:
        raise SystemExit("未找到训练权重")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "C组效果图"), exist_ok=True)

    model = YOLO(weights)
    names = model.names

    # ---------- 构建分组 ----------
    pos, neg = [], []
    for f in sorted(os.listdir(TEST_IMG)):
        if not f.lower().endswith(".jpg"):
            continue
        lbl = os.path.join(TEST_LBL, os.path.splitext(f)[0] + ".txt")
        has_ebike = False
        if os.path.exists(lbl):
            with open(lbl, "r", encoding="utf-8") as fh:
                for ln in fh:
                    if ln.strip().startswith("1"):   # 类别1 = 电动车
                        has_ebike = True
                        break
        (pos if has_ebike else neg).append(os.path.join(TEST_IMG, f))

    a_files = pos[:N_POS]
    b_files = neg
    c_files = [os.path.join(CROSS_DIR, f) for f in os.listdir(CROSS_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.isdir(CROSS_DIR) else []

    print(f"分组: A组(正样本)={len(a_files)}张  B组(负样本)={len(b_files)}张  C组(跨域)={len(c_files)}张")
    print(f"权重: {weights}")

    # ---------- 逐图推理一次(conf=0.10全量),三档阈值内存过滤 ----------
    rows = []          # CSV 明细
    perf = {}          # (组,阈值) -> [命中数, 总数, 置信度累计, 置信度个数]
    t0 = time.time()

    def judge(group, path, boxes, truth_ebike):
        """boxes: [(cls_id, conf), ...]"""
        for th in THRESHOLDS:
            key = (group, th)
            perf.setdefault(key, {"hit": 0, "n": 0, "conf_sum": 0.0, "conf_cnt": 0})
            ebike_boxes = [c for cid, c in boxes
                           if names[cid] == "ebike" and c >= th]
            p = perf[key]
            p["n"] += 1
            if ebike_boxes:
                p["hit"] += 1
                p["conf_sum"] += max(ebike_boxes)
                p["conf_cnt"] += 1
            rows.append({
                "组别": group, "文件": os.path.basename(path), "阈值": th,
                "真实含电动车": "是" if truth_ebike else "否",
                "检出电动车数": len(ebike_boxes),
                "最高置信度": round(max(ebike_boxes), 3) if ebike_boxes else "",
                "检出自行车数": sum(1 for cid, c in boxes
                                   if names[cid] == "bicycle" and c >= th),
            })

    def infer(group, files, truth_map=None):
        idx = 0
        for r in model.predict(source=files, conf=0.10, device=0,
                               imgsz=640, stream=True, verbose=False):
            path = files[idx]; idx += 1
            boxes = [(int(b.cls[0]), float(b.conf[0])) for b in r.boxes]
            truth = truth_map(path) if truth_map else None
            judge(group, path, boxes, truth)
            if group == "C":
                try:
                    r.save(filename=os.path.join(
                        OUT_DIR, "C组效果图", os.path.basename(path)))
                except Exception:
                    pass

    infer("A", a_files, lambda p: True)
    infer("B", b_files, lambda p: False)
    infer("C", c_files, lambda p: None)
    total_sec = time.time() - t0

    # ---------- 输出 CSV ----------
    csv_path = os.path.join(OUT_DIR, "robustness_detail.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- 汇总 markdown ----------
    def rate(k):
        p = perf[k]
        return p["hit"] / p["n"] * 100 if p["n"] else 0

    def avgc(k):
        p = perf[k]
        return p["conf_sum"] / p["conf_cnt"] * 100 if p["conf_cnt"] else 0

    lines = [
        "# 鲁棒性测试报告",
        "",
        f"- 权重: `{weights}`",
        f"- 测试时间: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- 总推理耗时: {total_sec:.1f}s (A{len(a_files)}+B{len(b_files)}+C{len(c_files)}张,每图仅推理一次)",
        "",
        "## 结果汇总",
        "",
        "| 组别 | 阈值 | 样本数 | 命中/检出数 | 比例 | 平均最高置信度 |",
        "|---|---|---|---|---|---|",
    ]
    for g, desc in [("A", "正样本·真值含电动车(检出率越高越好)"),
                     ("B", "负样本·真值无电动车(此比例即误报率,越低越好)"),
                     ("C", "跨域·陌生电梯实拍(参考观察)")]:
        for th in THRESHOLDS:
            k = (g, th)
            if perf.get(k, {}).get("n"):
                metric = "检出率" if g == "A" else ("误报率" if g == "B" else "检出比例")
                lines.append(f"| {g}组 {desc.split('·',1)[1]} | {th:.0%} | "
                             f"{perf[k]['n']} | {perf[k]['hit']} | "
                             f"{rate(k):.1f}% ({metric}) | {avgc(k):.1f}% |")
    lines += [
        "",
        "## 结论要点(按实测数据填写)",
        "",
        "1. A组检出率反映常规场景能力;对比三档阈值可看出阈值↑→检出率↓但误报率同步下降的权衡关系。",
        "2. B组误报率反映系统抗干扰能力(自行车/广告牌/行人不应触发电动车误报)。",
        "3. C组为完全陌生电梯环境的泛化参考,效果图见同目录 C组效果图/ 文件夹。",
        "",
        "> 明细数据: robustness_detail.csv (可用 Excel/WPS 打开)",
    ]
    md_path = os.path.join(OUT_DIR, "汇总报告.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n完成! 明细: {csv_path}")
    print(f"      汇总: {md_path}")
    for g in ("A", "B", "C"):
        for th in THRESHOLDS:
            k = (g, th)
            if perf.get(k, {}).get("n"):
                print(f"  {g}组 @{th:.0%}: {perf[k]['hit']}/{perf[k]['n']} = {rate(k):.1f}%")


if __name__ == "__main__":
    main()
