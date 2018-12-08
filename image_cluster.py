import numpy as np
import cv2
import dlib
from hdbscan import HDBSCAN
from facenet import facenet
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import normalize
from imutils import build_montages
import face_recognition as FR
from DistanceMetrics import Similarity


class ImageCluster(object):

    def __init__(self, imgs_path, blur=True, clusterSize=5, equalize=True, model='hog'):
        self.images = list()
        self.blur = blur
        self.equalize = equalize
        self.model = model
        self.clusterSize = clusterSize
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


    def _blur_check(self, image, threshold=75):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(image, cv2.CV_64F).var()
        if blur < threshold:
            return True  # Face is blurry!
        else:
            return False # Face is not blurry


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
                            image = cv2.resize(image, (int(w*0.4), int(h*0.4)))
                        print('[INFO] Processing image %d of %d; path : %s' % (i, len(self.images), path))

                        if self.equalize is True:
                            image = self._equalize(image)

                        boxes = FR.face_locations(image,
                                                  model=self.model,
                                                  number_of_times_to_upsample=2)
                        print('\t[INFO] Found %d faces' % (len(boxes)))
                        for box in boxes:
                            (t, r, b, l) = box
                            face = cv2.resize(image[t:b, l:r], (160, 160))
                            if self._blur_check(face) is True and self.blur is True:
                                print('\t[INFO] Skipping face - too blurry to process')
                                continue
                            #face = self._prewhiten(face)
                            feed_dict = {img_holder:[face], phase_train:False}
                            encoding = session.run(embeddings, feed_dict=feed_dict)

                            if len(encoding) > 0:
                                d = [{'path':path, 'loc':(t, r, b, l), 'encoding':encoding[0]}]
                                data.extend(d)

                    except:
                        print("There was an error. That's all we know.")
                        return False

        with open('data_points.pkl', 'wb') as file:
            pickle.dump(data, file)
            return True


    def cluster_data_points(self):
        with open('data_points.pkl', 'rb') as file:
            data = pickle.load(file)

        points = [d['encoding'] for d in data]
        points = np.vstack(points)
        points1 = normalize(points, norm='l2', axis=1)
        dist_metric = Similarity()
        VI = np.matmul(points1.T, points1)
        VI = np.linalg.inv(VI)

        clusterer = HDBSCAN(min_cluster_size=self.clusterSize,
                            metric='pyfunc',
                            func=dist_metric.fractional_distance)
        clusterer.fit(points1)
        results = list()


        #------------------------------- FOR TESTING ONLY -----------------------------------#

        labelIDs = np.unique(clusterer.labels_)
        print(labelIDs)
        for labelID in labelIDs:
            faces = list()
            idxs = np.where(clusterer.labels_ == labelID)[0]
            idxs = np.random.choice(idxs, size=min(64, len(idxs)),
                                    replace=False)
            for i in idxs:
                image = cv2.imread(data[i]['path'])
                (h, w) = image.shape[:2]
                if (h, w) > (640, 480):
                    image = cv2.resize(image, (int(w*0.4), int(h*0.4)))
                (t, r, b, l) = data[i]['loc']
                face = image[t:b, l:r]
                face = cv2.resize(face, (96, 96))
                faces.append(face)

            montage = build_montages(faces, (96, 96), (8, 8))[0]
            title = 'Face ID #{}'.format(labelID)
            title = 'Unknown Faces' if labelID == -1 else title
            cv2.imshow(title, montage)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('k'):
                idxs = np.where(clusterer.labels_ == labelID)[0]
                for i in idxs:
                    results.append(data[i]['path'])
                cv2.destroyAllWindows()
            elif key == ord('n'):
                cv2.destroyAllWindows()
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()

        return results

        #--------------------------------------------------------------------------------------#


if __name__ == '__main__':
    path = ""
    test = ImageCluster(path)
    test.create_data_points()
    test.cluster_data_points()
