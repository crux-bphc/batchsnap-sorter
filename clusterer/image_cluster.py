import argparse
import os
import pickle
import cv2
import dlib
import numpy as np
import tensorflow as tf
import face_recognition as FR
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from imutils.face_utils import FaceAligner
from facenet import facenet
from DistanceMetrics import Similarity
from multiprocessing import Queue, Process


MODEL = 'hog'
MIN_CLUSTER_SIZE = 5


def _equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l1 = clahe.apply(l)
    processed = cv2.merge((l1, a, b))
    return cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)


def _prewhiten(image):
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
    return np.multiply(np.subtract(image, mean), 1 / std_adj)


def _blur_check(image, threshold=50):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(image, cv2.CV_64F).var()
    return blur < threshold


def create_data_points(img_queue, enc_queue):
    predictor = dlib.shape_predictor('models/sp_68_point.dat')
    aligner = FaceAligner(predictor, desiredFaceWidth=300)

    with tf.Graph().as_default(), tf.Session() as session:
        graph = tf.get_default_graph()
        facenet.load_model('models/20180402-114759.pb')
        img_holder = graph.get_tensor_by_name('input:0')
        embeddings = graph.get_tensor_by_name('embeddings:0')
        phase_train = graph.get_tensor_by_name('phase_train:0')
        for path in iter(img_queue.get, 'END'):
            try:
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                (h, w) = image.shape[:2]
                if (w, h) > (640, 480):
                    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4,
                                       interpolation=cv2.INTER_AREA)
                image = _equalize(image)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                boxes = FR.face_locations(gray, model=MODEL,
                                          number_of_times_to_upsample=2)
                for box in boxes:
                    (t, r, b, l) = box
                    rect = dlib.rectangle(l, t, r, b)
                    face = aligner.align(image, gray, rect)
                    try:
                        y1, x2, y2, x1 = FR.face_locations(face, model=MODEL)[0]
                        face = cv2.resize(face[y1:y2, x1:x2], (160, 160))
                    except IndexError:
                        face = cv2.resize(image[t:b, l:r], (160, 160))
                    if _blur_check(face):
                        continue
                    face = _prewhiten(face)
                    feed_dict = {img_holder: [face], phase_train: False}
                    encoding = session.run(embeddings, feed_dict=feed_dict)
                    if len(encoding) > 0:
                        enc_queue.put({'path': path, 'encoding': encoding[0]})
            except Exception as e:
                enc_queue.put({'path': path, 'error': str(e)})


def cluster_data_points(data_points):
    points = [d['encoding'] for d in data_points]
    points = np.vstack(points)
    scaler = StandardScaler()
    scaler.fit(points)
    points = scaler.transform(points)
    dist_metric = Similarity()

    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                        metric='pyfunc',
                        func=dist_metric.fractional_distance)
    clusterer.fit(points)
    results = {}
    labelIDs = np.unique(clusterer.labels_)
    for labelID in labelIDs:
        paths = []
        encodings = []
        idxs = np.where(clusterer.labels_ == labelID)[0]
        for i in idxs:
            data = data_points[i]
            paths.append(data['path'])
            encodings.append(data['encoding'])
        results[labelID] = {
            'paths': paths,
            'mean_encoding': np.mean(np.asarray(encodings), axis=0),
            'std_dev': np.std(encodings, axis=0),
            'sample_size': len(paths)
        }
    return results


def gather_images(impath):
    for root, dirnames, files in os.walk(impath):
        for file in files:
            name = file.lower()
            if name.endswith('.jpg') or name.endswith('.png'):
                yield os.path.join(root, file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--impath", required=True,
                        help="Path to folder containing images to be sorted")
    args = parser.parse_args()
    inp_queue = Queue()
    enc_queue = Queue()
    NUM_PROCESSES = os.cpu_count() or 1
    for _ in range(NUM_PROCESSES):
        Process(target=create_data_points, args=(inp_queue, enc_queue)).start()
    for img in gather_images(args.impath):
        inp_queue.put(img)
    data_points = []
    while not inp_queue.empty():
        data = enc_queue.get()
        if 'error' in data:
            print('Errored', data)
        else:
            print('Processed', data['path'])
            data_points.append(data)
    for _ in range(NUM_PROCESSES):
        inp_queue.put('END')
    results = cluster_data_points(data_points)
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
