import os
import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from deepface import DeepFace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

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

def auto_correct_orientation(image):
    # Simplified rotation correction or use any other method
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

def process_image(cnic_image_url, selfie_image_url):
    cnic_image = download_image(cnic_image_url)
    selfie_image = download_image(selfie_image_url)
    rotated_cnic_image = auto_correct_orientation(cnic_image)
    cnic_temp_path = "temp_cnic.jpg"
    selfie_temp_path = "temp_selfie.jpg"
    cv2.imwrite(cnic_temp_path, rotated_cnic_image)
    cv2.imwrite(selfie_temp_path, selfie_image)
    try:
        result = DeepFace.verify(img1_path=cnic_temp_path, img2_path=selfie_temp_path)
        if result['verified']:
            analysis = DeepFace.analyze(img_path=selfie_temp_path, actions=['gender'])
            if isinstance(analysis, list):
                analysis = analysis[0]
            gender = analysis['gender']
            return f"Faces match. Detected gender: {gender}"
        else:
            return "Faces do not match"
    finally:
        os.remove(cnic_temp_path)
        os.remove(selfie_temp_path)

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
                "gender": "man" if "man" in result else "woman" if "woman" in result else None
            }
        })
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
