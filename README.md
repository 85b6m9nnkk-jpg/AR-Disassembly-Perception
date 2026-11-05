# AR-Disassembly-Perception

## Project Overview
This repository contains the core perception module for the Final Year Project: **Computer-Vision and AR Based Interactive Disassembly Guidance**. 

The module integrates traditional **YOLOv8 object detection** with a cutting-edge **Vision-Language Model (Qwen-VL)**. It is designed for End-of-Life (EOL) electronic products to achieve real-time part localization, tool recognition, and zero-shot classification of complex defects (e.g., rust, missing parts, cracked screens).

## Key Features
- **Dual-Camera Setup**: 
  - Camera 0 (Main View): Recognizes the part and its condition.
  - Camera 1 (Operator View): Detects the tool currently being used by the operator.
- **Local YOLOv8 Inference**: Provides millisecond-level latency for part and tool localization.
- **VLM-Assisted Reasoning**: Utilizes the Qwen-VL API to identify long-tail defects (e.g., `rust`, `cracked`, `burnt`) without the need for extensive retraining.
- **Poka-yoke (Error-Proofing) Mechanism**: Reads a local `rules.json` file to map conditions to expected tools. Triggers a visual WARNING if the operator picks up the wrong tool.

## Prerequisites
- Python 3.10
- Install the required dependencies:
```bash
pip install -r requirements.txt
```

(Note: You must configure a valid Qwen API Key in the code before running.)

##How to Run

1. Ensure both cameras are connected. (If Camera 0 and 1 are swapped, please exchange CAM_MAIN_ID and CAM_HAND_ID in the code).
2. Update rules.json to configure the mapping between defect conditions and recommended tools.
3. Run the main script:

```bash
python dual_camera_vlm.py
```

4. Press q to quit the camera windows.

##Author

· Zhang Jiayi (Perception & Interaction Module)
