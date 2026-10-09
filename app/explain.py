# -*- coding: utf-8 -*-
"""
============================================================
 可解释规则引擎 —— 电动车进电梯预警系统的"灵魂"模块
============================================================
职责: 把 YOLO 输出的原始数字(类别/置信度/坐标)翻译成
     人类能看懂的证据链 + 风险等级 + 处置建议。

设计理念(对应赛题"可解释性与人机交互"赛道):
  1. 每一次判定都给出"证据列表"——为什么报警、依据是什么
  2. 风险等级由透明规则推出,而非黑盒打分
  3. 阈值由用户掌控(人机交互),低于阈值的目标会显示为"已忽略"
============================================================
"""

# ---------- 风险等级定义 ----------
LEVELS = {
    "safe": {"cn": "安全", "color": "#10b981", "tag": "正常放行",
             "summary": "未检测到两轮车辆,轿厢状态正常"},
    "low":  {"cn": "低风险", "color": "#f59e0b", "tag": "提示引导",
             "summary": "检测到自行车(非锂电池车辆),允许进入,给予停放提示"},
    "high": {"cn": "高风险", "color": "#ef4444", "tag": "阻梯告警",
             "summary": "检测到电动自行车,触发阻梯告警,禁止本次运行"},
}

POLICY = ("依据《高层民用建筑消防安全管理规定》"
          "(应急管理部令第5号,2021年8月1日施行)第三十七条:"
          "禁止在高层民用建筑公共门厅、疏散走道、楼梯间、安全出口"
          "停放电动自行车或者为电动自行车充电。")

VOICE_HIGH = ("请注意,本电梯已检测到电动自行车。"
              "为了您和他人的安全,请勿将电动自行车推入电梯,"
              "请使用楼下集中充电区域。感谢您的配合。")


def conf_band(c):
    """置信度 → 人话分档"""
    if c >= 0.90:
        return "很高"
    if c >= 0.75:
        return "高"
    if c >= 0.55:
        return "中等"
    return "偏低"


def position_desc(cx, img_w):
    """目标中心 x 坐标 → 画面位置描述"""
    if cx < img_w / 3:
        return "画面左侧"
    if cx > img_w * 2 / 3:
        return "画面右侧"
    return "画面中部"


def area_desc(ratio):
    """检测框面积占画面比例 → 距离描述"""
    if ratio >= 0.20:
        return "目标很大、距离很近"
    if ratio >= 0.10:
        return "目标较大、距离较近"
    if ratio >= 0.04:
        return "目标中等、处于轿厢中部距离"
    return "目标较小、距离较远(可能在电梯门口)"


def area_text(ratio):
    """面积占比格式化:小于1%时避免显示为0%"""
    return f"{ratio:.1%}" if ratio < 0.01 else f"{ratio:.0%}"


def analyze(detections, persons, threshold, img_w, img_h):
    """
    核心规则引擎。
    参数:
      detections: [{cls:'ebike'/'bicycle', conf:0.93, box:[x1,y1,x2,y2], area_ratio, cx}, ...]
                  —— 后端以 conf=0.10 低阈值推理返回的全部目标
      persons:    [{conf, box, area_ratio, cx}, ...] —— 官方模型检出的人员
      threshold:  用户当前设定的判定阈值(滑块)
      img_w/img_h:图像宽高(像素)
    返回: 判定结果字典(等级/证据/建议/播报文案等)
    """
    # ---- 1. 按用户阈值筛选有效目标(体现"允许人类干预决策逻辑") ----
    active = [d for d in detections if d["conf"] >= threshold]
    ebikes = [d for d in active if d["cls"] == "ebike"]
    bicycles = [d for d in active if d["cls"] == "bicycle"]
    ignored = [d for d in detections if d["conf"] < threshold]

    # ---- 2. 风险等级判定(透明规则链) ----
    if len(ebikes) >= 1:
        level_key = "high"
    elif len(bicycles) >= 1:
        level_key = "low"
    else:
        level_key = "safe"
    level = LEVELS[level_key]

    # ---- 3. 证据链生成(聚合式表达:同类目标合并一条,避免逐条刷屏) ----
    evidence = []

    if ebikes:
        best = max(ebikes, key=lambda d: d["conf"])
        if len(ebikes) == 1:
            evidence.append({
                "icon": "ebike",
                "text": f"检测到 1 辆电动自行车,置信度 {best['conf']:.0%}({conf_band(best['conf'])}可信)",
            })
        else:
            evidence.append({
                "icon": "ebike",
                "text": (f"检测到 {len(ebikes)} 辆电动自行车"
                         f"(其中把握最高的一辆置信度 {best['conf']:.0%},{conf_band(best['conf'])}可信)"),
            })
        evidence.append({
            "icon": "position",
            "text": (f"主要目标位于{position_desc(best['cx'], img_w)},"
                     f"占画面约 {area_text(best['area_ratio'])} —— {area_desc(best['area_ratio'])}"),
        })

    if bicycles:
        bestb = max(bicycles, key=lambda d: d["conf"])
        if len(bicycles) == 1:
            evidence.append({
                "icon": "bicycle",
                "text": f"检测到 1 辆自行车,置信度 {bestb['conf']:.0%}({conf_band(bestb['conf'])}可信)——人力驱动,无锂电池",
            })
        else:
            evidence.append({
                "icon": "bicycle",
                "text": f"检测到 {len(bicycles)} 辆自行车(最高置信度 {bestb['conf']:.0%})——均为人力驱动,无锂电池",
            })

    if len(persons) >= 1:
        if level_key == "high":
            evidence.append({
                "icon": "person",
                "text": f"轿厢内同框检测到 {len(persons)} 名乘客 —— 人车密闭共存,"
                        f"锂电池热失控时无逃生空间,风险等级上调",
            })
        else:
            evidence.append({
                "icon": "person",
                "text": f"轿厢内检测到 {len(persons)} 名乘客",
            })

    if level_key == "high":
        evidence.append({
            "icon": "policy",
            "text": "判定依据:电动车搭载锂电池,轿厢为密闭狭小空间,"
                    "电池热失控可在数十秒内释放高温有毒烟气,并经电梯井形成烟囱效应",
        })

    if ignored and level_key != "high":
        evidence.append({
            "icon": "filter",
            "text": f"另有 {len(ignored)} 个疑似目标因置信度低于当前阈值({threshold:.0%})被忽略,"
                    f"可通过下方滑块降低阈值复核",
        })

    if level_key == "safe" and not persons and not detections:
        evidence.append({"icon": "shield",
                         "text": "画面内未检出两轮车辆,亦未检出人员,判定为空轿厢/无风险"})

    # ---- 4. 处置建议 ----
    advice = []
    if level_key == "high":
        advice = [
            {"title": "语音播报", "icon": "mic", "text": VOICE_HIGH},
            {"title": "联动梯控", "icon": "lift",
             "text": "保持轿厢门开启、暂停本次运行,直至电动车退出轿厢"},
            {"title": "物业告警", "icon": "bell",
             "text": "推送告警截图至物业值班端,留存违规证据,纳入楼栋安全档案"},
        ]
    elif level_key == "low":
        advice = [
            {"title": "语音提示", "icon": "mic",
             "text": "您好,自行车请靠边停放,注意不要占用轿厢空间,感谢配合。"},
        ]
    else:
        advice = [
            {"title": "正常放行", "icon": "shield",
             "text": "无违规目标,电梯正常运行;系统持续监测中"},
        ]

    # ---- 5. 汇总 ----
    return {
        "level": level_key,
        "level_cn": level["cn"],
        "level_tag": level["tag"],
        "level_color": level["color"],
        "summary": level["summary"],
        "counts": {
            "ebike": len(ebikes),
            "bicycle": len(bicycles),
            "person": len(persons),
            "ignored": len(ignored),
        },
        "threshold": threshold,
        "evidence": evidence,
        "advice": advice,
        "policy": POLICY,
        "voice": VOICE_HIGH if level_key == "high" else None,
    }
