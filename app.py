import os
import logging
import numpy as np
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilefacenet.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


def preprocess_image(image):
    # Resize and normalize the image
    image = cv2.resize(
        image, (input_details['shape'][1], input_details['shape'][2]))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32)  # Ensure type is float32
    return image


def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        img = Image.open(img_data).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
    except requests.RequestException as e:
        logger.error(f"Failed to download image from URL {url}: {e}")
        raise ValueError(f"Image at URL {url} could not be downloaded.")


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
    rotated_image = cv2.warpAffine(
        image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return rotated_image


def process_image(cnic_image_url, selfie_image_url):
    cnic_image = download_image(cnic_image_url)
    selfie_image = download_image(selfie_image_url)

    # Correct orientation of CNIC image
    cnic_image = auto_correct_orientation(cnic_image)

    # Preprocess images
    cnic_image = preprocess_image(cnic_image)
    selfie_image = preprocess_image(selfie_image)

    # Perform face verification using TFLite model
    interpreter.set_tensor(input_details['index'], cnic_image)
    interpreter.invoke()
    cnic_embedding = interpreter.get_tensor(output_details['index'])[0]

    interpreter.set_tensor(input_details['index'], selfie_image)
    interpreter.invoke()
    selfie_embedding = interpreter.get_tensor(output_details['index'])[0]

    # Debug: Check embeddings before normalization
    logger.info(f"CNIC embedding (pre-norm): {cnic_embedding}")
    logger.info(f"Selfie embedding (pre-norm): {selfie_embedding}")

    # Normalize embeddings
    cnic_embedding = cnic_embedding / np.linalg.norm(cnic_embedding)
    selfie_embedding = selfie_embedding / np.linalg.norm(selfie_embedding)

    # Debug: Check embeddings after normalization
    logger.info(f"CNIC embedding (post-norm): {cnic_embedding}")
    logger.info(f"Selfie embedding (post-norm): {selfie_embedding}")

    # Calculate similarity
    similarity = cosine_similarity([cnic_embedding], [selfie_embedding])[0][0]
    logger.info(f"Similarity score: {similarity}")

    # Use a more appropriate threshold based on model performance
    threshold = 0.5  # Adjust as needed

    if similarity > threshold:
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
