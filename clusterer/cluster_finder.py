import cv2
from clusterer.facenet import facenet
import pickle
import numpy as np
from clusterer.DistanceMetrics import Similarity
import face_recognition as FR
import tensorflow as tf


def _equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l1 = clahe.apply(l)
    processed = cv2.merge((l1, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)

    return processed


tf.Graph().as_default()
session = tf.Session()

facenet.load_model('clusterer/models/20180402-114759.pb')
img_holder = tf.get_default_graph().get_tensor_by_name(
    'input:0')
embeddings = tf.get_default_graph().get_tensor_by_name(
    'embeddings:0')
phase_train = tf.get_default_graph().get_tensor_by_name(
    'phase_train:0')


def _prewhiten(image):
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
    processed = np.multiply(np.subtract(image, mean), 1 / std_adj)

    return processed


def prepare_encodings(image):
    representative = None
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _equalize(image)
    faces = FR.face_locations(image,
                              model='hog')
    if len(faces) == 1:
        (t, r, b, l) = faces[0]
        try:
            face = cv2.resize(image[t:b, l:r], (160, 160))
            face = _prewhiten(face)
            feed_dict = {img_holder: [face], phase_train: False}
            encoding = session.run(embeddings, feed_dict=feed_dict)
            if len(encoding) == 0:
                pass
            else:
                representative = encoding[0]
        except:
            print(
                'Could not process face; make sure that the face is actually detectable')
            return None
    return representative


def find_clusters(representative=None, use_CI=True, sigma=1.25):
    with open('results.pkl', 'rb') as file:
        data = pickle.load(file)
    metric = Similarity()
    possibilities = list()
    if representative is not None:
        if use_CI is True:
            # Use confidence intervals to check whether the person corresponds
            # to a given cluster or not. A z-distribution is assumed here.
            for labelID in data.keys():
                if labelID == -1:
                    pass
                mean_encoding = data[labelID]['mean_encoding']
                error = data[labelID]['std_dev']
                n = data[labelID]['sample_size']
                lower_bound = mean_encoding - \
                    (np.multiply(error, 1.96) / np.power(n, 0.5))
                upper_bound = mean_encoding + \
                    (np.multiply(error, 1.96) / np.power(n, 0.5))
                l1 = representative >= lower_bound
                l2 = representative <= upper_bound
                if np.all(l1 & l2):
                    result = {'labelID': labelID,
                              'paths': data[labelID]['paths']}
                    possibilities.append(result)
        else:
            for labelID in data.keys():
                centre_point = data[labelID]['mean_encoding']
                error = data[labelID]['std_dev'] * sigma
                sphere_point = np.add(centre_point, error)
                sphere_radius = metric.fractional_distance(
                    centre_point, sphere_point)
                distance = metric.fractional_distance(
                    centre_point, representative)
                if distance <= sphere_radius and labelID != -1:
                    result = {'labelID': labelID,
                              'paths': data[labelID]['paths']}
                    possibilities.append(result)

    return possibilities


def get_images(image):
    encodings = prepare_encodings(image)
    if encodings is None:
        return []
    clusters = find_clusters(encodings)
    if len(clusters) == 1:
        return clusters[0]['paths']
    else:
        lo, hi = 0.5, 2
        for _ in range(20):
            mid = (lo + hi) / 2
            clusters = find_clusters(encodings, False, mid)
            if len(clusters) == 1:
                return clusters[0]['paths']
            elif not clusters:
                lo = mid
            else:
                hi = mid
    return []
