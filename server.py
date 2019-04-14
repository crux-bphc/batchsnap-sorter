#!/usr/bin/env python3
import os
import tempfile
from flask import Flask, send_from_directory, abort, jsonify, request
from clusterer.cluster_finder import get_images
import cv2


BUILD_FOLD = os.environ.get('BUILD_FOLD', 'frontend/build/')
IMAGES_FOLD = os.environ.get('IMAGES_FOLD', '/images')

app = Flask("Batchsnap Sorter", static_folder=BUILD_FOLD)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def index():
    return send_from_directory(BUILD_FOLD, 'index.html')


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
        links = [('/images/' + f) for f in links]
    return jsonify({'links': links})


@app.route('/images/<filename>')
def serve_images(filename):
    return send_from_directory(IMAGES_FOLD, filename)


@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(BUILD_FOLD, path)):
        return send_from_directory(BUILD_FOLD, path)
    abort(404)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
