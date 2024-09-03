import os
import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import normalize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load TensorFlow Lite model
model_path = 'mobilefacenet.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Define TensorFlow Lite model input/output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
    except requests.RequestException as e:
        logger.error(f"Failed to download image from URL {url}: {e}")
        raise ValueError(f"Image at URL {url} could not be downloaded.")

def preprocess_image(image):
    image = cv2.resize(image, (112, 112))  # Resize to MobileFaceNet input size
    image = image.astype(np.float32)
    image = preprocess_input(image)  # MobileNet preprocessing
    return image

def get_embedding(image):
    preprocessed_image = preprocess_image(image)
    input_data = np.expand_dims(preprocessed_image, axis=0)
    
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    embeddings = interpreter.get_tensor(output_details['index'])
    return embeddings.flatten()

def auto_correct_orientation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return rotated_image

def cosine_similarity(emb1, emb2):
    emb1 = normalize([emb1])[0]
    emb2 = normalize([emb2])[0]
    return np.dot(emb1, emb2)

def process_image(cnic_image_url, selfie_image_url):
    cnic_image = download_image(cnic_image_url)
    selfie_image = download_image(selfie_image_url)
    rotated_cnic_image = auto_correct_orientation(cnic_image)
    
    cnic_embedding = get_embedding(rotated_cnic_image)
    selfie_embedding = get_embedding(selfie_image)

    similarity = cosine_similarity(cnic_embedding, selfie_embedding)
    
    # Define a threshold for face match (you might need to tune this)
    threshold = 0.6
    
    if similarity >= threshold:
        return f"Faces match. Similarity score: {similarity:.2f}"
    else:
        return f"Faces do not match. Similarity score: {similarity:.2f}"

@app.route('/match', methods=['POST'])
def match_cnic_selfie():
    data = request.json
    cnic_image_url = data.get('cnic_image_url')
    selfie_image_url = data.get('selfie_image_url')

    if not cnic_image_url or not selfie_image_url:
        return jsonify({"error": "CNIC image URL and selfie image URL are required"}), 400

    try:
        result = process_image(cnic_image_url, selfie_image_url)
        return jsonify({
            "res": {
                "matched": "Faces match" in result,
                "similarity_score": result
            }
        })
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
