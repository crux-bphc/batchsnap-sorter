import argparse
import logging
import os
import pickle
import queue
import cv2
import dlib
import numpy as np
import tensorflow as tf
import face_recognition as FR
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from imutils.face_utils import FaceAligner
from DistanceMetrics import Similarity
from multiprocessing import Queue, Process, current_process

logging.basicConfig(filename='clusterer.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s')
MODEL = 'hog'


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


def _blur_check(image, threshold=20):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(image, cv2.CV_64F).var()
    return blur < threshold

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

def create_data_points(img_queue, enc_queue):
    predictor = dlib.shape_predictor('models/sp_68_point.dat')
    aligner = FaceAligner(predictor, desiredFaceWidth=300)
    proc_name = current_process().name

    with tf.Graph().as_default(), tf.Session() as session:
        graph = tf.get_default_graph()
        load_model('models/triplet_loss.pb')
        img_holder = graph.get_tensor_by_name('input:0')
        embeddings = graph.get_tensor_by_name('embeddings:0')
        phase_train = graph.get_tensor_by_name('phase_train:0')
        while True:
            try:
                path = img_queue.get(timeout=300)
            except queue.Empty:
                logging.info('Input queue is empty. Shutting down process.')
                break
            else:
                if path == 'END':
                    break
            logging.debug("{}: Processing {}".format(proc_name, path))
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
                        y1, x2, y2, x1 = FR.face_locations(
                            face, model=MODEL)[0]
                        face = cv2.resize(face[y1:y2, x1:x2], (160, 160))
                    except IndexError:
                        face = cv2.resize(image[t:b, l:r], (160, 160))
                    # if _blur_check(face):
                    #     logging.debug(f"{proc_name}: Face too blurry")
                    #     continue
                    face = _prewhiten(face)
                    feed_dict = {img_holder: [face], phase_train: False}
                    encoding = session.run(embeddings, feed_dict=feed_dict)
                    if len(encoding) > 0:
                        enc_queue.put({'path': path, 'encoding': encoding[0]})
            except Exception as e:
                enc_queue.put({'path': path, 'error': str(e)})


def cluster_data_points(data_points, cluster_size=5, distance_metric_func="Euclidean"):
    points = [d['encoding'] for d in data_points]
    points = np.vstack(points)
    scaler = StandardScaler()
    scaler.fit(points)
    points = scaler.transform(points)
    dist_metric = Similarity()
    if distance_metric_func == "Fractional":
        dist_metric_func = dist_metric.fractional_distance
    else:
        dist_metric_func = dist_metric.euclidean_distance
    clusterer = HDBSCAN(min_cluster_size=cluster_size,
                        metric='pyfunc',
                        func=dist_metric_func)
    clusterer.fit(points)
    logging.info("Fit complete.")
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
    logging.info(f"Number of clusters: {len(results)}")
    return results


def gather_images(impath):
    try:
        with open('data_points.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        processed_paths = set()
    else:
        processed_paths = {d['path'] for d in data}
    logging.debug("Already processed {} images.".format(len(processed_paths)))
    for root, dirnames, files in os.walk(impath):
        for file in files:
            name = file.lower()
            if name.endswith('.jpg') or name.endswith('.png'):
                path = os.path.join(root, file)
                if path not in processed_paths:
                    yield path


def run_feature_detection(args):
    if args.clean and not args.path:
        print("Please provide path to images folder")
        return
    inp_queue = Queue()
    enc_queue = Queue()
    NUM_PROCESSES = args.cores or 1  # in case os.cpu_count() returns None
    logging.info(f"Starting {NUM_PROCESSES} processes.")
    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=create_data_points, args=(inp_queue, enc_queue))
        p.name = str(i)
        p.start()
        processes.append(p)
    logging.info("Gathering images")
    for count, img in enumerate(gather_images(args.path)):
        inp_queue.put(img)
    logging.debug(f"Put {count} images for processing.")
    if args.clean:
        data_points = []
        try:
            os.renames('data_points.pkl', 'data_points.pkl.old')
            os.renames('results.pkl', 'results.pkl.old')
        except FileNotFoundError:
            pass
    else:
        try:
            with open('data_points.pkl', 'rb') as f:
                data_points = pickle.load(f)
        except FileNotFoundError:
            data_points = []
    count = 0
    while True:
        if inp_queue.empty():
            logging.info("Input queue is empty. Finishing up.")
            if enc_queue.empty():
                break
        try:
            data = enc_queue.get(timeout=600)
        except queue.Empty:
            logging.error("No processing is happening for some reason.")
            break
        if 'error' in data:
            logging.error('Errored ' + data['error'])
        else:
            data_points.append(data)
            logging.debug('Found encoding in ' + data['path'])
            count += 1
            if count % 5:
                continue
            with open('data_points.pkl', 'wb') as f:
                pickle.dump(data_points, f, protocol=pickle.HIGHEST_PROTOCOL)

    for _ in range(NUM_PROCESSES):
        inp_queue.put('END')
    logging.info("Finished feature detection.")

    logging.info("Killing remaining processes.")
    for i, p in enumerate(processes):
        if p.is_alive():
            logging.debug(f"Killed process {i}.")
            p.terminate()


def run_clustering(args):
    logging.debug("args: " + str(vars(args)))
    with open('data_points.pkl', 'rb') as f:
        data_points = pickle.load(f)
    logging.debug(f"Data points in file: {len(data_points)}")
    results = cluster_data_points(
        data_points, args.cluster_size, args.distance_metric)
    with open(f'results_{args.cluster_size}_{args.distance_metric}.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(results))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_fd = subparsers.add_parser('fd')
    parser_fd.add_argument("--path",
                           help="Path to folder containing images to be sorted")
    parser_fd.add_argument('--clean-start', action='store_true', dest='clean',
                           help="Discard results of previous runs and start over")
    parser_fd.add_argument('--cores', type=int, default=os.cpu_count(),
                           help="Number of cores to use during feature detection")
    parser_fd.set_defaults(func=run_feature_detection)
    parser_cl = subparsers.add_parser('cl')
    parser_cl.add_argument('-s', '--cluster-size', type=int, default=5,
                           help="Minimum number of images to form a cluster")
    parser_cl.add_argument('-d', '--distance-metric',
                           choices=["Fractional", "Euclidean"], default="Euclidean",
                           help="Distance metric to be used for the clusterer")
    parser_cl.set_defaults(func=run_clustering)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
