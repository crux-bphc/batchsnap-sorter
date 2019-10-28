import numpy as np
import cv2
import dlib
from hdbscan import HDBSCAN
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import normalize
from shutil import copy
import argparse
import face_recognition as FR
from DistanceMetrics import Similarity

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

class VideoCluster(object):

    def __init__(self, vid_path, blur=1, clusterSize=2, equalize=1, model='hog', sortpath="./Results"):
        SUPPORTED_EXTENSIONS = ['.mp4', '.3gp', '.mkv', '.avi']
        self.videos = list()
        self.blur = True if blur==1 else False
        self.equalize = True if equalize==1 else False
        if model == 'hog' or model == 'cnn':
            self.model = model
        else:
            self.model = 'hog'
        self.clusterSize = clusterSize if clusterSize > 0 else 5
        self.cs = self.clusterSize
        self.sortPath = sortpath
        if not os.path.exists(vid_path):
            print('The specified image folder path does not exist.')
            exit(0)
        for root, dirnames, files in os.walk(vid_path):
            for video in files:
                for extension in SUPPORTED_EXTENSIONS:
                    if video.lower().endswith(extension):
                        self.videos.append(os.path.join(root, video))
        self.videos.sort()
        print('Found %d videos!' % (len(self.videos)))


    def _equalize(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l1 = clahe.apply(l)
        processed = cv2.merge((l1, a, b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)

        return processed


    def _prewhiten(self, image):
        mean = np.mean(image)
        std = np.std(image)
        std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
        processed = np.multiply(np.subtract(image, mean), 1/std_adj)
        
        return processed


    def _blur_check(self, image, threshold=75):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(image, cv2.CV_64F).var()
        if blur < threshold:
            return True  # Face is blurry!
        else:
            return False # Face is not blurry


    def _compute_statistics(self, encodings):
        mean_encoding = np.mean(np.asarray(encodings), axis=0)
        std_dev = np.std(encodings, axis=0)

        return (mean_encoding, std_dev)


    def create_data_points(self):
        results = list()
        with tf.Graph().as_default():
            with tf.Session() as session:

                load_model('models/triplet_loss.pb')
                img_holder = tf.get_default_graph().get_tensor_by_name(
                                    'input:0')
                embeddings = tf.get_default_graph().get_tensor_by_name(
                                    'embeddings:0')
                phase_train = tf.get_default_graph().get_tensor_by_name(
                                    'phase_train:0')

                for (i, path) in enumerate(self.videos, 1):
                    try:
                        data = list()
                        capture = cv2.VideoCapture(path)
                        print('[INFO] Processing video %d of %d; path : %s' % (i, len(self.videos), path))
                        total_frames = capture.get(7)
                        print('\t[INFO] Total number of frames : %d' % (total_frames))
                        self.clusterSize = int(total_frames/10) if total_frames > 30 else 5
                        get_next = True
                        for n in range(int(total_frames)-1):
                            if get_next is True:
                                ret, frame = capture.read()
                                if not ret:
                                    print('\tError!')
                                    continue
                                get_next = not get_next
                                try:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                except:
                                    pass
                                (h, w) = frame.shape[:2]
                                if (h, w) > (640, 480):
                                    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
                                if self.equalize is True:
                                    frame = self._equalize(frame)
                                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                                boxes = FR.face_locations(gray,
                                                          model=self.model,
                                                          number_of_times_to_upsample=2)
                                for box in boxes:
                                    (t, r, b, l) = box
                                    face = cv2.resize(frame[t:b, l:r], (160, 160))
                                    if self.blur is True:
                                        if self._blur_check(face) is True:
                                            print('\t[INFO] Skipping face - too blurry to process')
                                            continue
                                    face = self._prewhiten(face)
                                    feed_dict = {img_holder:[face], phase_train:False}
                                    encoding = session.run(embeddings, feed_dict=feed_dict)

                                    if len(encoding) > 0:
                                        d = [{'path':path, 'encoding':encoding[0]}]
                                        data.extend(d)
                                else:
                                    get_next = True
                        net_faces = self.cluster_data_points(data, processed=False)
                        if net_faces is not None:
                            for labelID in net_faces.keys():
                                d = [{'path':net_faces[labelID]['paths'][0], 'encoding':net_faces[labelID]['mean_encoding']}]
                                results.extend(d)


                    except:
                        print("There was an error. That's all we know.")
                        return False

        with open('video_data.pkl', 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        return True


    def cluster_data_points(self, data=None, processed=False):
        if data is None or len(data) < 1:
            return None
        if processed is True:
            with open('video_data.pkl', 'rb') as file:
                data = pickle.load(file)
                self.clusterSize = self.cs
        points = [d['encoding'] for d in data]
        points = np.vstack(points)
        points = normalize(points, norm='l2', axis=1)
        dist_metric = Similarity()

        clusterer = HDBSCAN(min_cluster_size=self.clusterSize,
                            metric='pyfunc',
                            func=dist_metric.fractional_distance)
        clusterer.fit(points)
        results = dict()

        labelIDs = np.unique(clusterer.labels_)
        for labelID in labelIDs:
            idxs = np.where(clusterer.labels_ == labelID)[0]
            encodings = list()
            for i in idxs:
                if labelID not in results:
                    results[labelID] = dict()
                    results[labelID]['paths'] = list()
                    results[labelID]['mean_encoding'] = None
                    results[labelID]['std_dev'] = None
                results[labelID]['paths'].append(data[i]['path'])
                encodings.append(data[i]['encoding'])
            results[labelID]['mean_encoding'], results[labelID]['std_dev'] = self._compute_statistics(encodings)

        if processed is False:
            return results
        else:
            with open('video_results.pkl', 'wb') as file:
                pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

            return results

        return None


    def sort_videos(self):
        with open('video_results.pkl', 'rb') as file:
            data = pickle.load(file)

        if not os.path.exists(self.sortPath):
            os.mkdir(self.sortPath)

        for labelID in data.keys():
            if labelID == -1:
                name = os.path.join(self.sortPath, 'Unknown')
            else:
                name = os.path.join(self.sortPath, str(labelID))
            pathlist = data[labelID]['paths']
            if not os.path.exists(name):
                os.mkdir(name)
            for path in pathlist:
                copy(path, name)

        print('Done')
        return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vidpath", required=True,
                        help="Path to folder containing videos to be sorted")
    parser.add_argument("-s", "--sortpath", required=False, default="./Results",
                        help="Path to folder where sorted videos will be saved. Default is ./Results")
    parser.add_argument("-b", "--blurcheck", required=False, default=1,
                        help="Perform a quality check using blurriness of frame. Default is 1 (check for blur)")
    parser.add_argument("-e", "--equalize", required=False, default=1,
                        help="Equalize lighting conditions before processing. Default is 1 (equalize lighting)")
    parser.add_argument("-c", "--clustersize", required=False, default=2,
                        help="Minimum number of videos expected per person. Default is 2")
    parser.add_argument("-m", "--model", required=False, default="hog",
                        help="Face detection model. Can be 'hog' or 'cnn'. Using 'cnn' is more accurate, but slower. Default is 'hog'")
    arguments = vars(parser.parse_args())
    cluster = VideoCluster(vid_path=arguments['vidpath'],
                           blur=arguments['blurcheck'],
                           clusterSize=arguments['clustersize'],
                           equalize=arguments['equalize'],
                           model=arguments['model'],
                           sortpath=arguments['sortpath'])
    cluster.create_data_points()
    cluster.cluster_data_points(processed=True)
    cluster.sort_videos()



