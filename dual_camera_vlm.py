import cv2
import json
import base64
import requests
import re
import time
from ultralytics import YOLO
from datetime import datetime

# ================= 配置区 =================
# YOLO 模型路径
PART_MODEL_PATH = r"C:\Users\1\runs\detect\train-6\weights\best.pt"
TOOL_MODEL_PATH = r"C:\Users\1\runs\detect\train-12\weights\best.pt"

# 千问 VLM 配置
API_KEY = "YOUR_QWEN_API_KEY_HERE"
VLM_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1/chat/completions"

CONF_THRESHOLD_PART = 0.6
CONF_THRESHOLD_TOOL = 0.6


CAM_MAIN_ID = 1  # 对着桌面的主视角 (如果还是反的，把这里改回 0)
CAM_HAND_ID = 0  # 拍手的视角 (如果还是反的，把这里改回 1)

# ================= 核心函数 =================
def call_qwen_vlm(cropped_img):
    """调用千问大模型分析裁剪后的图像"""
    _, buffer = cv2.imencode('.jpg', cropped_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    prompt = """
    You are an expert in industrial defect detection. Analyze this component image.
    Classify into ONE category: 'normal', 'rust', 'missing', or 'wear'.
    Respond strictly in valid JSON format WITHOUT Markdown:
    {"condition": "rust/wear/missing/normal", "severity": "mild/moderate/severe", "reason": "brief explanation"}
    """
    
    payload = {
        "model": "qwen3-vl-plus",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        "temperature": 0.1
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(VLM_URL, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            clean_content = re.sub(r'```json|```', '', content).strip()
            return json.loads(clean_content)
    except Exception as e:
        print(f"VLM 调用失败/超时: {e}")
    return None

# ================= 初始化 =================
print("正在加载 YOLO 模型...")
model_part = YOLO(PART_MODEL_PATH)
model_tool = YOLO(TOOL_MODEL_PATH)

cap_main = cv2.VideoCapture(CAM_MAIN_ID)
cap_hand = cv2.VideoCapture(CAM_HAND_ID)

if not cap_main.isOpened() or not cap_hand.isOpened():
    print("错误：无法打开摄像头，请检查摄像头是否被其他程序占用。")
    exit()

print("系统启动！按 'q' 退出。")

# 缓存变量（防止每帧都去请求 API 导致卡顿）
last_vlm_time = 0
cached_condition = "none"
cached_reason = ""
cached_severity = "none"

while True:
    ret_main, frame_main = cap_main.read()
    ret_hand, frame_hand = cap_hand.read()
    if not ret_main or not ret_hand: break

    # ================= Camera 0：零件与状态 =================
    detected_part = "none"
    crop_bbox = None
    
    results_part = model_part(frame_main, verbose=False)
    for box in results_part[0].boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD_PART: continue
        cls_name = model_part.names[int(box.cls[0])]
        if cls_name == "background": continue
            
        crop_bbox = box.xyxy[0].tolist()
        detected_part = cls_name
        x1, y1, x2, y2 = map(int, crop_bbox)
        cv2.rectangle(frame_main, (x1, y1), (x2, y2), (0, 255, 0), 2)
        break

    # ================= VLM 推理 (时间节流) =================
    if crop_bbox is not None and (time.time() - last_vlm_time > 2.0):
        x1, y1, x2, y2 = map(int, crop_bbox)
        crop_img = frame_main[max(0, y1-10):min(frame_main.shape[0], y2+10), 
                              max(0, x1-10):min(frame_main.shape[1], x2+10)]
        
        if crop_img.size > 0:
            print("正在请求 VLM 评估状态...")
            vlm_result = call_qwen_vlm(crop_img)
            if vlm_result:
                cached_condition = vlm_result.get("condition", "normal")
                cached_severity = vlm_result.get("severity", "none")
                cached_reason = vlm_result.get("reason", "")
                print(f"VLM 判定: {cached_condition} ({cached_severity})")
            last_vlm_time = time.time()

    # ================= Camera 1：手部与工具 =================
    detected_tool = "none"
    results_tool = model_tool(frame_hand, verbose=False)
    for box in results_tool[0].boxes:
        conf = float(box.conf[0])
        if conf > CONF_THRESHOLD_TOOL:
            detected_tool = model_tool.names[int(box.cls[0])]
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_hand, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_hand, f"Tool: {detected_tool}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            break

    # ================= 核心逻辑：读取外部规则 =================
    system_status = "IDLE"
    recommendation = ""
    
    if detected_part != "none" and detected_tool != "none" and cached_condition != "none":
        try:
            with open(r"C:\Users\1\Desktop\rules.json", "r", encoding="utf-8") as f:
                rules_dict = json.load(f)
        except:
            rules_dict = {"wear": "screwdriver", "rust": "hammer", "normal": "any"} 
            
        expected_tool = rules_dict.get(cached_condition, "any")
        
        if expected_tool == "any" or detected_tool == expected_tool:
            system_status = "OK"
            recommendation = f"Proceed with {detected_tool}"
        else:
            system_status = "WARNING"
            recommendation = f"Wrong tool! Please use {expected_tool}"

    # 在主画面上绘制文字
    color_main = (0, 0, 255) if cached_condition in ["rust", "wear", "missing"] else (0, 255, 0)
    cv2.putText(frame_main, f"Condition: {cached_condition.upper()} ({cached_severity})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_main, 2)
    
    if system_status == "WARNING":
        cv2.putText(frame_main, recommendation, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif system_status == "OK":
        cv2.putText(frame_main, recommendation, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ================= 组装 JSON =================
    output = {
        "timestamp": datetime.now().isoformat(),
        "sensing": {
            "part_name": detected_part,
            "condition": cached_condition,
            "severity": cached_severity,
            "vlm_reason": cached_reason
        },
        "interaction": {
            "detected_tool": detected_tool
        },
        "system_feedback": {
            "status": system_status,
            "message": recommendation
        }
    }

    try:
        with open(r"C:\Users\1\Desktop\dual_camera_output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except PermissionError:
        pass

    cv2.imshow("Cam 0 - VLM Condition & Feedback", frame_main)
    cv2.imshow("Cam 1 - Tool", frame_hand)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_main.release()
cap_hand.release()
cv2.destroyAllWindows()