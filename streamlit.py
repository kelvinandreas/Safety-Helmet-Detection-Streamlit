import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from datetime import datetime
import time
import glob

st.set_page_config(
    page_title="Smart Vision"
)

# Load YOLO model
model = YOLO('./model.pt')

# Folder untuk menyimpan tangkapan pelanggaran
capture_folder = 'captures'
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

# Fungsi untuk menghapus semua file di folder capture
def clear_captures(folder):
    files = glob.glob(os.path.join(folder, '*'))
    for f in files:
        os.remove(f)

# Template images
template_images = {
    "Template 1": "./template/template1.jpg",
    "Template 2": "./template/template2.jpg",
    "Template 3": "./template/template3.jpg",
    "Template 4": "./template/template4.jpeg",
    "Template 5": "./template/template5.jpg",
    "Template 6": "./template/template6.jpg",
    "Template 7": "./template/template7.jpg",
    "Template 8": "./template/template8.jpg",
    "Template 9": "./template/template9.jpg",
    "Template 10": "./template/template10.jpg",
    "Template 11": "./template/template11.jpg",
}

# Fungsi untuk prediksi dan menggambar kotak deteksi
def predict_and_draw(frame):
    results = model(frame)
    boxes = results[0].boxes
    alert = False
    captured_image = None
    no_helmet_detected = False

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()  # Ambil rasio kepercayaan model (confidence score)

        if conf >= 0.3:  # Hanya plot jika confidence >= 75%
            if cls == 0 and not no_helmet_detected:  # Helm terdeteksi
                label = f"PASS ({conf*100:.1f}%)"
                color = (0, 255, 0)  # Green
            elif cls == 1:  # Tidak memakai helm
                label = f"No Helmet ({conf*100:.1f}%)"
                color = (255, 0, 0)  # Red
                alert = True
                captured_image = frame[y1:y2, x1:x2]
                no_helmet_detected = True  # Prioritas jika deteksi tidak memakai helm

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, alert, captured_image

# Judul dan menu navigasi
logo = Image.open('./logo.jpg')
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=300)
st.title("Smart Vision👁️")

menu = st.selectbox("Select Mode", ["Webcam", "Upload Image", "Use Template"])

# Definisi awal variabel penghapusan periodik
last_clear_time = time.time()

if menu == "Webcam":
    FRAME_WINDOW = st.image([])
    captured_images_container = st.container()

    cap = cv2.VideoCapture(0)
    last_capture_time = 0
    capture_interval = 15  # seconds
    captured_images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_frame, alert, captured_image = predict_and_draw(frame_rgb)

        FRAME_WINDOW.image(result_frame)

        # if alert and captured_image is not None:
        #     current_time = time.time()
        #     if current_time - last_capture_time >= capture_interval:
        #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #         capture_path = os.path.join(capture_folder, f"no_helmet_{timestamp}.png")
        #         cv2.imwrite(capture_path, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
        #         captured_images.append(capture_path)
        #         last_capture_time = current_time

        # with captured_images_container:
        #     if captured_images:
        #         cols = st.columns(len(captured_images))
        #         for i, img_path in enumerate(captured_images):
        #             with cols[i]:
        #                 st.image(img_path, caption=f"Captured: {os.path.basename(img_path)}", use_container_width=True)

        # Hapus file setiap 1 menit
        # if time.time() - last_clear_time >= 60:
        #     clear_captures(capture_folder)
        #     captured_images.clear()
        #     last_clear_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif menu == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        result_image, alert, _ = predict_and_draw(image_np)

        with col2:
            st.image(result_image, caption="Detection Result", use_container_width=True)

elif menu == "Use Template":
    selected_template = st.selectbox("Or choose a template image", list(template_images.keys()))

    if selected_template:
        template_path = template_images[selected_template]
        image = Image.open(template_path)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Template Image", use_container_width=True)

        image_np = np.array(image)
        result_image, alert, _ = predict_and_draw(image_np)

        with col2:
            st.image(result_image, caption="Detection Result", use_container_width=True)
