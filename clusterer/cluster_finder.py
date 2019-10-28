import cv2
import pickle
import numpy as np
from DistanceMetrics import Similarity
import face_recognition as FR
import tensorflow as tf
import os

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

tf.Graph().as_default()
session = tf.Session()

load_model('models/triplet_loss.pb')
img_holder = tf.get_default_graph().get_tensor_by_name(
    'input:0')
embeddings = tf.get_default_graph().get_tensor_by_name(
    'embeddings:0')
phase_train = tf.get_default_graph().get_tensor_by_name(
    'phase_train:0')

results_file = os.environ.get('RESULTS_FILE', 'results.pkl')
try:
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
except (FileNotFoundError, pickle.PickleError):
    print("Unable to load clustering results.")
    exit(1)


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
    processed = np.multiply(np.subtract(image, mean), 1 / std_adj)

    return processed


def prepare_encodings(image):
    representative = None
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _equalize(image)
    faces = FR.face_locations(image,
                              model='hog')
    if len(faces) != 1:
        return None
    (t, r, b, l) = faces[0]
    try:
        face = cv2.resize(image[t:b, l:r], (160, 160))
        face = _prewhiten(face)
        feed_dict = {img_holder: [face], phase_train: False}
        encoding = session.run(embeddings, feed_dict=feed_dict)
        if len(encoding) != 0:
            representative = encoding[0]
    except:
        return None
    return representative


def find_clusters(representative, use_CI=True, sigma=1.25):
    metric = Similarity()
    possibilities = list()
    if use_CI:
        # Use confidence intervals to check whether the person corresponds
        # to a given cluster or not. A z-distribution is assumed here.
        for labelID, cluster in data.items():
            if labelID == -1:
                continue
            mean_encoding = cluster['mean_encoding']
            error = cluster['std_dev']
            n = cluster['sample_size']
            lower_bound = mean_encoding - \
                (np.multiply(error, 1.96) / np.power(n, 0.5))
            upper_bound = mean_encoding + \
                (np.multiply(error, 1.96) / np.power(n, 0.5))
            l1 = representative >= lower_bound
            l2 = representative <= upper_bound
            if np.all(l1 & l2):
                result = {'labelID': labelID,
                          'paths': cluster['paths']}
                possibilities.append(result)
    else:
        for labelID, cluster in data.items():
            if labelID == -1:
                continue
            centre_point = cluster['mean_encoding']
            error = cluster['std_dev'] * sigma
            sphere_point = np.add(centre_point, error)
            sphere_radius = metric.fractional_distance(centre_point, sphere_point)
            distance = metric.fractional_distance(centre_point, representative)
            if distance <= sphere_radius:
                result = {'labelID': labelID, 'paths': cluster['paths']}
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
