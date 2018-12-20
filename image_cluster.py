import numpy as np
import cv2
import dlib
from hdbscan import HDBSCAN
from facenet import facenet
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import normalize
from shutil import copy
import argparse
import face_recognition as FR
from DistanceMetrics import Similarity


class ImageCluster(object):

    def __init__(self, imgs_path, blur=1, clusterSize=5, equalize=1, model='hog', sortpath="./Results"):
        self.images = list()
        self.blur = True if blur==1 else False
        self.equalize = True if equalize==1 else False
        if model == 'hog' or model == 'cnn':
            self.model = model
        else:
            self.model = 'hog'
        self.clusterSize = clusterSize if clusterSize > 0 else 5
        self.sortPath = sortpath
        if not os.path.exists(imgs_path):
            print('The specified image folder path does not exist.')
            exit(0)
        for root, dirnames, files in os.walk(imgs_path):
            for image in files:
                if image.lower().endswith('.jpg') or image.lower().endswith('.png'):
                    self.images.append(os.path.join(root, image))
        self.images.sort()


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


    def _blur_check(self, image, threshold=50):
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
        data = list()
        with tf.Graph().as_default():
            with tf.Session() as session:

                facenet.load_model('models/20180402-114759.pb')
                img_holder = tf.get_default_graph().get_tensor_by_name(
                                    'input:0')
                embeddings = tf.get_default_graph().get_tensor_by_name(
                                    'embeddings:0')
                phase_train = tf.get_default_graph().get_tensor_by_name(
                                    'phase_train:0')

                for (i, path) in enumerate(self.images, 1):
                    try:
                        image = cv2.imread(path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        (h, w) = image.shape[:2]
                        if (h, w) > (640, 480):
                            image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
                        print('[INFO] Processing image %d of %d; path : %s' % (i, len(self.images), path))

                        if self.equalize is True:
                            image = self._equalize(image)

                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                        boxes = FR.face_locations(gray,
                                                  model=self.model,
                                                  number_of_times_to_upsample=2)
                        print('\t[INFO] Found %d faces' % (len(boxes)))
                        for box in boxes:
                            (t, r, b, l) = box
                            face = cv2.resize(image[t:b, l:r], (160, 160))
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

                    except:
                        print("There was an error. That's all we know.")
                        return False

        with open('data_points.pkl', 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            return True


    def cluster_data_points(self):
        with open('data_points.pkl', 'rb') as file:
            data = pickle.load(file)

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

        with open('results.pkl', 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        return True


    def sort_images(self):
        with open('results.pkl', 'rb') as file:
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
    parser.add_argument("-i", "--impath", required=True,
                        help="Path to folder containing images to be sorted")
    parser.add_argument("-s", "--sortpath", required=False, default="./Results",
                        help="Path to folder where sorted images will be saved. Default is ./Results")
    parser.add_argument("-b", "--blurcheck", required=False, default=1,
                        help="Perform a quality check using blurriness of image. Default is 1 (check for blur)")
    parser.add_argument("-e", "--equalize", required=False, default=1,
                        help="Equalize lighting conditions before processing. Default is 1 (equalize lighting)")
    parser.add_argument("-c", "--clustersize", required=False, default=5,
                        help="Minimum number of images expected per person. Default is 5")
    parser.add_argument("-m", "--model", required=False, default="hog",
                        help="Face detection model. Can be 'hog' or 'cnn'. Using 'cnn' is more accurate, but slower. Default is 'hog'")
    arguments = vars(parser.parse_args())
    cluster = ImageCluster(imgs_path=arguments['impath'],
                           blur=arguments['blurcheck'],
                           clusterSize=arguments['clustersize'],
                           equalize=arguments['equalize'],
                           model=arguments['model'],
                           sortpath=arguments['sortpath'])
    cluster.create_data_points()
    cluster.cluster_data_points()
    cluster.sort_images()
