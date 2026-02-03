
import os
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "weights/best.pt"

model = YOLO(MODEL_PATH)

def check_helmet_image(image):
    if image is None:
        return None, "Vui lòng tải ảnh lên."

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_frame = frame.copy()

    results = model(frame, imgsz=640, conf=0.4)

    has_helmet = False
    no_helmet = False
    helmet_count = 0
    violation_count = 0

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy
        class_ids = results[0].boxes.cls
        confs = results[0].boxes.conf

        for box, cls, conf in zip(boxes, class_ids, confs):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cls = int(cls)

            if cls == 1:  # WithHelmet
                has_helmet = True
                helmet_count += 1
                color = (0, 255, 0)
                label = f"Doi mu {conf:.2f}"

            elif cls == 2:  # WithoutHelmet
                no_helmet = True
                violation_count += 1
                color = (0, 0, 255)
                label = f"Khong doi mu {conf:.2f}"

            else:
                continue

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

    if no_helmet and has_helmet:
        status = f"PHÁT HIỆN HỖN HỢP:\n{helmet_count} đội mũ\n{violation_count} không đội mũ"
    elif no_helmet:
        status = f"VI PHẠM: {violation_count} người không đội mũ"
    elif has_helmet:
        status = f"AN TOÀN: {helmet_count} người đội mũ"
    else:
        status = "Không phát hiện người."

    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return annotated_frame, status


with gr.Blocks(theme=gr.themes.Soft(), title="Helmet Detection System") as demo:
    gr.Markdown("""
    # Hệ Thống Kiểm Tra Mũ Bảo Hiểm
Kiểm tra người có đội mũ bảo hiểm hay không từ ảnh
    """)
    
    gr.Markdown("### Tải ảnh lên để kiểm tra người có đội mũ bảo hiểm hay không")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Ảnh đầu vào", type="numpy")
            image_btn = gr.Button("Kiểm Tra", variant="primary", size="lg")
        
        with gr.Column():
            image_output = gr.Image(label="Kết quả phát hiện")
            image_status = gr.Textbox(label="Trạng thái", lines=3, interactive=False)
    
    image_btn.click(
        fn=check_helmet_image,
        inputs=image_input,
        outputs=[image_output, image_status]
    )
    
    gr.Markdown("""
    ---
    ### Hướng dẫn:
    - Tải ảnh lên và nhấn **Kiểm Tra** để xem kết quả
    - **Khung xanh**: Người đội mũ bảo hiểm
    - **Khung đỏ**: Người không đội mũ bảo hiểm
    """)


demo.launch(share=True)