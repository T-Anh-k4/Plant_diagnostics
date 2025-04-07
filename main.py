import os
import numpy as np
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# Cấu hình thư mục uploads
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load mô hình bệnh cây
plant_model = load_model('model (4).h5')
plant_labels = {0: 'Bacteria', 1: 'Fungi', 2: 'Healthy', 3: 'Pests', 4: 'Virus'}

# Cấu hình Gemini
try:
    genai.configure(api_key="AIzaSyDrtctq6rPpqPxLJGP6IVbhboVUpru7ZM0")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    print(f"Lỗi khi khởi tạo Gemini: {str(e)}")
    model = None


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    predictions = plant_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return plant_labels[predicted_class]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def ask_ai(message):
    try:
        if model is None:
            return "Hệ thống AI đang bảo trì. Vui lòng thử lại sau."

        response = model.generate_content(
            {
                "parts": [{"text": f"""Bạn là chuyên gia nông nghiệp. Hãy trả lời ngắn gọn, chính xác bằng tiếng Việt.
                          Câu hỏi: {message}"""}],
            },
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 500,
                "top_p": 0.9
            }
        )

        return response.text if hasattr(response, "text") else "Không thể hiểu phản hồi từ AI."

    except Exception as e:
        print("Lỗi khi gọi Gemini:", e)
        return f"Hiện không thể kết nối với AI. Lỗi: {str(e)}"



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    f = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)
    predicted_label = predict_disease(file_path)
    return str(predicted_label)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Thiếu dữ liệu đầu vào"}), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({"error": "Tin nhắn không được trống"}), 400

        response = ask_ai(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({
            "error": f"Lỗi hệ thống: {str(e)}",
            "response": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
