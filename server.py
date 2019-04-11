#!/usr/bin/env python3
import os
from flask import Flask, send_from_directory, abort

BUILD_FOLD = 'frontend/build/'
app = Flask("Batchsnap Sorter", static_folder=BUILD_FOLD)


@app.route('/')
def index():
    return send_from_directory(BUILD_FOLD, 'index.html')


@app.route('/image', methods=['POST'])
def handle_image():
    return "Coming soon"


@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(BUILD_FOLD, path)):
        return send_from_directory(BUILD_FOLD, path)
    abort(404)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
