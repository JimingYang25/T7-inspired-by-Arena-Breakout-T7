import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# ---------- 模型配置 ----------
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}
encoder = 'vits' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 加载模型 ----------
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load('./checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device).eval()

# ---------- 摄像头参数 ----------
camera_id = 0          # 默认摄像头，外接USB可能为1或2
frame_width = 640      # 降低分辨率以提升帧率
frame_height = 480
skip_frames = 0        # 每N帧推理一次，0表示每帧都推理（若卡顿可调大）

# ---------- 打开摄像头 ----------
cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()

print("按 'q' 或 ESC 退出")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    frame_count += 1
    
    if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
        
        cv2.imshow("Depth Anything V2 - Real-time", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        continue

    # ---------- 模型推理 ----------
    # 将BGR转为RGB（模型要求RGB）
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推理（返回深度图，numpy数组，尺寸与原图相同）
    with torch.no_grad():
        depth = model.infer_image(rgb)  # 返回 HxW 的深度值（float32）

    # ---------- 后处理与可视化 ----------
    # 归一化到0~255用于显示
    depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
   
    combined = np.hstack((frame, depth))
    cv2.imshow("Depth Anything V2 - Real-time", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()