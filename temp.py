import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import logging
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình thư mục uploads
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình bệnh cây
plant_model = load_model('model (4).h5')
plant_labels = {0: 'Bacteria', 1: 'Fungi', 2: 'Healthy', 3: 'Pests', 4: 'Virus'}

# Load mô hình chatbot
try:
    chatbot_model = load_model('chatbot_model (3).h5')
    logger.info("Đã tải thành công mô hình chatbot")
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình chatbot: {str(e)}")
    chatbot_model = None

# Tải dữ liệu từ file training_data.pkl
try:
    with open('training_data (1).pkl', 'rb') as file:
        data = pickle.load(file)
        words = data['words']
        classes = data['classes']
    logger.info("Đã tải thành công dữ liệu training")
except Exception as e:
    logger.error(f"Lỗi khi tải dữ liệu training: {str(e)}")
    words = []
    classes = []

# Đọc intents với mã hóa UTF-8
try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    logger.info("Đã tải thành công file intents")
except Exception as e:
    logger.error(f"Lỗi khi tải file intents: {str(e)}")
    intents = {'intents': []}

# Khởi tạo lemmatizer
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo NLP: {str(e)}")
    lemmatizer = None


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    predictions = plant_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return plant_labels[predicted_class]


# Hàm tiền xử lý câu hỏi đầu vào
def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu: {str(e)}")
        return []


# Chuyển câu hỏi thành vector Bag of Words
def bow(sentence, words):
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)
    except Exception as e:
        logger.error(f"Lỗi khi tạo BoW: {str(e)}")
        return np.zeros(len(words))


# Dự đoán nhãn cho câu hỏi
def predict_class(sentence):
    try:
        bow_vector = bow(sentence, words)
        prediction = chatbot_model.predict(np.array([bow_vector]))[0]
        ERROR_THRESHOLD = 0.7
        predicted_classes = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]

        # Sắp xếp theo độ tin cậy giảm dần
        predicted_classes.sort(key=lambda x: x[1], reverse=True)

        return predicted_classes
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán lớp: {str(e)}")
        return []


# Lấy câu trả lời từ intents
def get_response(predicted_classes, intents_json):
    try:
        if not predicted_classes:
            return random.choice([
                "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?",
                "Tôi chưa học về vấn đề này. Bạn muốn hỏi về bệnh cây cụ thể nào?",
                "Câu hỏi của bạn về bệnh cây phải không? Hãy mô tả chi tiết hơn."
            ])

        top_class = predicted_classes[0][0]
        intent_tag = classes[top_class]

        for intent in intents_json['intents']:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])

        return "Tôi không tìm thấy thông tin phù hợp cho câu hỏi của bạn."
    except Exception as e:
        logger.error(f"Lỗi khi lấy phản hồi: {str(e)}")
        return "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn."


def ask_ai(message):
    predicted_classes = predict_class(message)
    response = get_response(predicted_classes, intents)
    return response


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

        if not chatbot_model:
            return jsonify({"error": "Mô hình chatbot chưa sẵn sàng"}), 500

        response = ask_ai(user_message)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Lỗi khi xử lý chat: {str(e)}")
        return jsonify({
            "error": f"Lỗi hệ thống: {str(e)}",
            "response": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
