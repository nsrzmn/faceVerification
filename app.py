# app.py
from flask import Flask, request, jsonify
import cv2
import pytesseract
from pytesseract import Output
from deepface import DeepFace
import numpy as np
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Utility functions (same as in your original script)


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
    else:
        raise ValueError(f"Image at URL {url} could not be downloaded.")


def auto_correct_orientation(image):
    data = pytesseract.image_to_osd(image, output_type=Output.DICT)
    angle = data.get('rotate', 0)
    angle = angle if angle <= 180 else angle - 360

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    abs_cos = abs(np.cos(np.radians(angle)))
    abs_sin = abs(np.sin(np.radians(angle)))
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    M[0, 2] += (bound_w - w) / 2
    M[1, 2] += (bound_h - h) / 2

    rotated_image = cv2.warpAffine(
        image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC)

    return rotated_image


def process_image(cnic_image_url, selfie_image_url):
    cnic_image = download_image(cnic_image_url)
    selfie_image = download_image(selfie_image_url)

    rotated_cnic_image = auto_correct_orientation(cnic_image)

    cnic_temp_path = "temp_cnic.jpg"
    selfie_temp_path = "temp_selfie.jpg"
    cv2.imwrite(cnic_temp_path, rotated_cnic_image)
    cv2.imwrite(selfie_temp_path, selfie_image)

    result = DeepFace.verify(img1_path=cnic_temp_path,
                             img2_path=selfie_temp_path)

    if result['verified']:
        analysis = DeepFace.analyze(
            img_path=selfie_temp_path, actions=['gender'])
        if isinstance(analysis, list):
            analysis = analysis[0]
        gender = analysis['gender']
        result_text = f"Faces match. Detected gender: {gender}"
    else:
        result_text = "Faces do not match"

    return result_text

# Flask route to handle image processing


@app.route('/process_images', methods=['POST'])
def process_images():
    data = request.json
    cnic_image_url = data.get('cnic_image_url')
    selfie_image_url = data.get('selfie_image_url')

    if not cnic_image_url or not selfie_image_url:
        return jsonify({"error": "Both CNIC and selfie image URLs are required."}), 400

    try:
        result = process_image(cnic_image_url, selfie_image_url)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
