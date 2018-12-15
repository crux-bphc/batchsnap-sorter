import cv2
from facenet import facenet
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from DistanceMetrics import Similarity
import face_recognition as FR
import tensorflow as tf
from math import sqrt


class PhotoClusterFinder(object):

    def __init__(self, sortpath='./Results'):
        self.sortPath = sortpath

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

    def prepare_encodings(self):
        representative = None

        '''This function involves capturing about 5 images of a person through
           the webcam and computing the mean encoding for that face. Since this
           application is best deployed on a remote server, I am not sure about
           how to write this code, as it is meant to be run on the client side.
           On top of that, I am not sure whether I should save this mean encoding
           on the client side or on the server side. Therefore, this code will
           be written as if this whole class will be loaded on the client side;
           this will be used for testing purposes for now.'''

        with tf.Graph().as_default():
            with tf.Session() as session:

                facenet.load_model('models/20180402-114759.pb')
                img_holder = tf.get_default_graph().get_tensor_by_name(
                                    'input:0')
                embeddings = tf.get_default_graph().get_tensor_by_name(
                                'embeddings:0')
                phase_train = tf.get_default_graph().get_tensor_by_name(
                                'phase_train:0')
                
                capture = cv2.VideoCapture(0)
                count = 0
                encodings = list()
                while True:
                    ret, frame = capture.read()
                    if not ret:
                        continue
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = self._equalize(image)
                    faces = FR.face_locations(image,
                                              model='hog')
                    text = 'Count : ' + str(count)
                    for (t, r, b, l) in faces:
                        cv2.rectangle(frame, (l, t),
                                      (r, b),
                                      (0, 255, 0), 2)
                        y = t - 10 if t - 10 > 10 else t + 10
                        cv2.putText(frame, text,
                                    (l, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    cv2.imshow('Webcam Feed', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('k'):
                        try:
                            face = cv2.resize(image[t:b, l:r], (160, 160))
                            face = self._prewhiten(face)
                            feed_dict = {img_holder:[face], phase_train:False}
                            encoding = session.run(embeddings, feed_dict=feed_dict)
                            if len(encoding) == 0:
                                pass
                            else:
                                encodings.append(encoding[0])
                                count += 1
                        except:
                            pass
                    elif key == ord('q') and count >= 5:
                        encodings = np.asarray(encodings)
                        representative = np.mean(encodings, axis=0)
                        break

        #------------- FOR DEBUG ----------------#
        with open('representative.pkl', 'wb') as file:
            pickle.dump(representative, file)
        #----------------------------------------#

        return representative

    def find_clusters(self, representative=None):
        with open('results.pkl', 'rb') as file:
            data = pickle.load(file)
        #------------ FOR DEBUG -----------------#
        with open('representative.pkl', 'rb') as file:
            representative = pickle.load(file)
        #----------------------------------------#
        metric = Similarity()
        possibilities = list()
        if representative is not None:
            representative = normalize(np.asarray([representative]), norm='l2', axis=1)
            for labelID in data.keys():
                centre_point = data[labelID]['mean_encoding']
                error = data[labelID]['std_dev']*1.50 
                # We assume that our representative encoding lies within 1.5 standard
                # deviations of the mean for it to match that cluster
                sphere_point = np.add(centre_point, error)
                sphere_radius = metric.fractional_distance(centre_point, sphere_point)
                distance = metric.fractional_distance(centre_point, representative)
                if distance <= sphere_radius and labelID != -1:
                    possibilities.append(labelID)

        return possibilities


if __name__ == '__main__':
    pcf = PhotoClusterFinder()
    representative = pcf.prepare_encodings()
    clusters = pcf.find_clusters()

    print(clusters)
