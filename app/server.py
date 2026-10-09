# -*- coding: utf-8 -*-
"""
============================================================
 ElevatorGuard · 电动车进电梯智能阻车预警系统 —— 本地服务器(开源版)
============================================================
 技术选型(刻意零第三方Web框架):
   - Python 标准库 http.server 搭建,不引入 flask/gradio 等任何新依赖,
     保证深度学习环境(torch/cv2)零污染、现场离线可用
   - 前端为原生 HTML/CSS/JS 单文件,见 web/index.html

 启动: 双击 start.bat (或 python server.py)
 访问: http://127.0.0.1:8300

 路径说明: 自动优先使用仓库自带 weights/ 下的模型;
   找不到时回退到本机训练目录,方便二次训练后无缝切换。
============================================================
"""
import base64
import glob
import json
import os
import random
import socket
import sys
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2
import torch
from ultralytics import YOLO

import explain  # 同目录的规则引擎

# ==================== 配置区 ====================
PORT = int(os.environ.get("EG_PORT", 8300))        # 端口可用环境变量 EG_PORT 覆盖
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
REPO_ROOT = os.path.dirname(ROOT)                  # 仓库根目录
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")   # 仓库自带权重
SAMPLE_DIR = os.path.join(ROOT, "samples")         # 演示示例图(可选)
LOCAL_RUNS_DIR = r"E:\mytrain\runs"                # 本机训练目录(回退用)
LOW_CONF = 0.10      # 推理采用低阈值全量检出,由前端滑块决定"判定阈值"
PERSON_CONF = 0.35   # 人员检出的固定阈值
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 自动选择 GPU/CPU
# ===============================================


def resolve_ebike_weights():
    """优先仓库自带权重, 回退本机训练目录中最新 best.pt"""
    bundled = os.path.join(WEIGHTS_DIR, "best.pt")
    if os.path.exists(bundled):
        return bundled
    cands = glob.glob(os.path.join(LOCAL_RUNS_DIR, "*", "weights", "best.pt"))
    return max(cands, key=os.path.getmtime) if cands else None


def resolve_person_weights():
    bundled = os.path.join(WEIGHTS_DIR, "yolo11n.pt")
    if os.path.exists(bundled):
        return bundled
    return "yolo11n.pt"   # ultralytics 会自动下载官方权重(需联网)


def trained_epochs(weights_path):
    """读取 results.csv 推算该权重的训练轮数(答辩时可展示)"""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(weights_path)), "results.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            return max(len(lines) - 1, 0)
    except Exception:
        pass
    return None


# ---------- 全局模型(启动时加载一次) ----------
print("=" * 60)
print("ElevatorGuard · 电动车进电梯智能阻车预警系统 · 正在加载模型...", flush=True)

EBIKE_WEIGHTS = resolve_ebike_weights()
if EBIKE_WEIGHTS is None:
    print("[错误] 未找到训练权重! 请将 best.pt 放到仓库 weights/ 文件夹")
    sys.exit(1)
PERSON_WEIGHTS = resolve_person_weights()

_t0 = time.time()
ebike_model = YOLO(EBIKE_WEIGHTS)     # 自训模型: 0=bicycle 1=ebike
person_model = YOLO(PERSON_WEIGHTS)   # 官方模型: COCO 80类(只用 person)
EBIKE_NAMES = ebike_model.names
LOAD_SECONDS = time.time() - _t0
EPOCHS = trained_epochs(EBIKE_WEIGHTS)

print(f"  车辆模型: {EBIKE_WEIGHTS}")
print(f"           类别表: {EBIKE_NAMES}  (已训 {EPOCHS} 轮)")
print(f"  人员模型: {PERSON_WEIGHTS} (COCO 官方预训练)")
print(f"  推理设备: {DEVICE}  模型加载耗时 {LOAD_SECONDS:.1f}s, 服务端口 {PORT}")
print("=" * 60, flush=True)

# ---------- 推理预热(避免首次请求因冷启动慢) ----------
print("推理预热中...", flush=True)
try:
    _warm = np.zeros((480, 640, 3), dtype=np.uint8)
    ebike_model.predict(_warm, conf=0.10, device=DEVICE, imgsz=640, verbose=False)
    person_model.predict(_warm, conf=PERSON_CONF, classes=[0], device=DEVICE,
                         imgsz=640, verbose=False)
    print("预热完成,首次请求将秒级响应", flush=True)
except Exception as e:
    print(f"预热跳过({e})——不影响使用,首次请求会稍慢", flush=True)


def run_detection(img_bgr):
    """双模型推理: 自训模型检车(低阈值全量) + 官方模型检人"""
    h, w = img_bgr.shape[:2]

    r1 = ebike_model.predict(img_bgr, conf=LOW_CONF, device=DEVICE,
                             imgsz=640, verbose=False)[0]
    r2 = person_model.predict(img_bgr, conf=PERSON_CONF, classes=[0],
                              device=DEVICE, imgsz=640, verbose=False)[0]

    detections = []
    for b in r1.boxes:
        cls_id = int(b.cls[0])
        cls_name = EBIKE_NAMES[cls_id] if cls_id < len(EBIKE_NAMES) else str(cls_id)
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)
        detections.append({
            "cls": cls_name,
            "conf": float(b.conf[0]),
            "box": [x1, y1, x2, y2],
            "area_ratio": area_ratio,
            "cx": (x1 + x2) / 2.0,
        })

    persons = []
    for b in r2.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        persons.append({
            "conf": float(b.conf[0]),
            "box": [x1, y1, x2, y2],
            "area_ratio": max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h),
            "cx": (x1 + x2) / 2.0,
        })

    return detections, persons, w, h


class Handler(BaseHTTPRequestHandler):
    """单线程请求处理(本地单用户场景,避免模型并发冲突)"""

    # ---- 基础响应工具 ----
    def _send(self, code, content, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 30 * 1024 * 1024:
            return None
        return self.rfile.read(n)

    def log_message(self, fmt, *args):  # 精简日志
        pass

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json({
                    "ok": True,
                    "ebike_weights": EBIKE_WEIGHTS,
                    "ebike_names": EBIKE_NAMES,
                    "epochs": EPOCHS,
                    "person_weights": PERSON_WEIGHTS,
                    "device": str(DEVICE),
                    "load_seconds": round(LOAD_SECONDS, 1),
                    "server": "ElevatorGuard 零依赖本地服务 v1.1 (开源版)",
                })
            elif path == "/api/sample":
                if not (os.path.isdir(SAMPLE_DIR)):
                    self._json({"ok": False,
                                "msg": "未配置示例图目录: 可在 app/samples/ 放入"
                                       "任意电梯图片后重启(或直接用页面'上传图片')"}, 500)
                    return
                files = [f for f in os.listdir(SAMPLE_DIR)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if not files:
                    self._json({"ok": False, "msg": "示例图目录为空"}, 500)
                    return
                fname = random.choice(files)
                with open(os.path.join(SAMPLE_DIR, fname), "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                self._json({"ok": True, "image": "data:image/jpeg;base64," + b64,
                            "filename": fname})
            else:
                self._json({"ok": False, "msg": "not found"}, 404)
        except Exception as e:
            self._json({"ok": False, "msg": f"{type(e).__name__}: {e}"}, 500)

    # ---- POST ----
    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            raw = self._read_body()
            if raw is None:
                self._json({"ok": False, "msg": "请求体为空或超过30MB"}, 400)
                return
            body = json.loads(raw.decode("utf-8"))

            if path == "/api/detect":
                # 上传检测: image(base64 dataURL) + threshold(判定阈值)
                t0 = time.time()
                b64 = body.get("image", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    self._json({"ok": False, "msg": "图片解码失败,请更换图片"}, 400)
                    return
                threshold = float(body.get("threshold", 0.35))

                detections, persons, w, h = run_detection(img)
                verdict = explain.analyze(detections, persons, threshold, w, h)
                verdict["detections"] = detections
                verdict["persons"] = persons
                verdict["img_size"] = [w, h]
                verdict["infer_ms"] = int((time.time() - t0) * 1000)
                print(f"[检测] 判定={verdict['level_cn']} 电动车x{verdict['counts']['ebike']} "
                      f"人x{verdict['counts']['person']} 耗时{verdict['infer_ms']}ms", flush=True)
                self._json({"ok": True, "verdict": verdict})

            elif path == "/api/analyze":
                # 阈值滑块重判: 只跑规则引擎,不重复推理(毫秒级)
                detections = body.get("detections", [])
                persons = body.get("persons", [])
                threshold = float(body.get("threshold", 0.35))
                w, h = body.get("img_size", [640, 640])
                verdict = explain.analyze(detections, persons, threshold, w, h)
                # 回传全量目标,供前端画布重绘灰虚线框(低于阈值目标)
                verdict["detections"] = detections
                verdict["persons"] = persons
                verdict["img_size"] = [w, h]
                self._json({"ok": True, "verdict": verdict})

            else:
                self._json({"ok": False, "msg": "not found"}, 404)
        except Exception as e:
            self._json({"ok": False, "msg": f"{type(e).__name__}: {e}"}, 500)


def get_lan_ip():
    """获取真实局域网IP(优先192.168/10.x,跳过VPN虚拟网卡198.18.x和169.254链路本地)"""
    ips = []
    try:
        # 方法1: 通过hostname枚举
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        # 方法2: 连外网测出口IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip not in ips:
            ips.append(ip)
    except Exception:
        pass
    # 过滤: 去掉VPN虚拟段(198.18.x是RFC保留段)和链路本地
    bad = ("198.18.", "169.254.")
    ips = [ip for ip in ips if not ip.startswith(bad)]
    # 优先返回 192.168.x 或 10.x(真实WiFi/局域网)
    for ip in ips:
        if ip.startswith(("192.168.", "10.")):
            return ip
    return ips[0] if ips else "127.0.0.1"


def find_openssl():
    """查找可用的 OpenSSL(优先 yolo 环境自带)"""
    cands = [
        os.path.join(sys.prefix, "Library", "bin", "openssl.exe"),
        r"D:\anaconda3\envs\yolo\Library\bin\openssl.exe",
        r"D:\anaconda3\Library\bin\openssl.exe",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def generate_ssl_cert(ip, cert_dir):
    """为当前局域网 IP 生成自签名证书(手机 HTTPS 摄像头用)"""
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    cnf_file = os.path.join(cert_dir, "openssl_san.cnf")

    # 写 SAN 配置(包含当前 IP + localhost)
    with open(cnf_file, "w", encoding="utf-8") as f:
        f.write(f"""[req]
distinguished_name = req_dn
x509_extensions = v3_req
prompt = no

[req_dn]
CN = ElevatorGuard

[v3_req]
subjectAltName = IP:{ip}, IP:127.0.0.1, DNS:localhost
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
""")

    openssl = find_openssl()
    if not openssl:
        return None, None

    import subprocess
    try:
        subprocess.run(
            [openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256",
             "-keyout", key_file, "-out", cert_file,
             "-days", "30", "-nodes", "-config", cnf_file],
            capture_output=True, timeout=15)
        if os.path.exists(cert_file) and os.path.exists(key_file):
            return cert_file, key_file
    except Exception:
        pass
    return None, None


def start_https_server(cert_file, key_file, https_port):
    """启动 HTTPS 服务(手机摄像头需要安全上下文)"""
    try:
        import ssl
        import threading
        https_server = HTTPServer(("0.0.0.0", https_port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_file, key_file)
        https_server.socket = ctx.wrap_socket(https_server.socket, server_side=True)
        t = threading.Thread(target=https_server.serve_forever, daemon=True)
        t.start()
        return True
    except Exception as e:
        print(f"  HTTPS 启动失败: {e}", flush=True)
        return False


def main():
    lan_ip = get_lan_ip()
    url = f"http://127.0.0.1:{PORT}"
    lan_url = f"http://{lan_ip}:{PORT}"
    https_port = PORT + 1  # HTTPS 用 8301(HTTP 8300 + 1)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)  # 启动后自动打开浏览器
    # 绑定 0.0.0.0: 手机等局域网设备可通过 http://本机IP:8300 访问
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n服务已启动 → 本机访问 {url}", flush=True)
    print(f"             手机访问 {lan_url} (需同一WiFi/热点)", flush=True)

    # HTTPS 服务(手机摄像头 getUserMedia 需要安全上下文)
    cert_file, key_file = generate_ssl_cert(lan_ip, ROOT)
    if cert_file:
        ok = start_https_server(cert_file, key_file, https_port)
        if ok:
            https_url = f"https://{lan_ip}:{https_port}"
            print(f"             手机摄像头 {https_url} (首次点\"高级→继续访问\")", flush=True)
            print(f"             摄像头端口防火墙也需放行(管理员执行):", flush=True)
            print(f'             netsh advfirewall firewall add rule name="ElevatorGuardHTTPS" dir=in action=allow protocol=TCP localport={https_port}', flush=True)

    print(f"             (Ctrl+C 停止)", flush=True)
    print(f"\n  手机无法访问时检查Windows防火墙(管理员PowerShell执行):", flush=True)
    print(f'  netsh advfirewall firewall add rule name="ElevatorGuard" dir=in action=allow protocol=TCP localport={PORT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
