#!/usr/bin/env python3
import tempfile
from flask import Flask, abort, jsonify, request
from clusterer.cluster_finder import get_images
import cv2


app = Flask("Batchsnap Sorter")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/image/', methods=['POST'])
def handle_image():
    if 'image' not in request.files:
        abort(400)
    img = request.files['image']
    if img.content_type not in ('image/jpeg', 'image/png'):
        abort(400)
    img_type = '.png' if img.content_type.endswith('png') else '.jpg'
    with tempfile.NamedTemporaryFile(suffix=img_type) as temp:
        img.save(temp)
        image = cv2.imread(temp.name)
        links = get_images(image)
    return jsonify({'links': links})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
