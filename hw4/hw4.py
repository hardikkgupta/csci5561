import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from scipy import stats
from pathlib import Path, PureWindowsPath
import os
from tqdm import tqdm

def get_tiny_image(img, tiny_size):
    def resize_image(img, new_size):
        height, width = img.shape
        new_height, new_width = new_size
        y_scale = height / new_height
        x_scale = width / new_width

        new_image = np.zeros((new_height, new_width), dtype=np.uint8)

        for i in range(new_height):
            for j in range(new_width):
                y = int(i * y_scale)
                x = int(j * x_scale)

                y_frac = i * y_scale - y
                x_frac = j * x_scale - x

                y = max(0, min(y, height - 2))
                x = max(0, min(x, width - 2))
                new_image[i, j] = (1 - y_frac) * ((1 - x_frac) * img[y, x] + x_frac * img[y, x + 1]) + \
                                  y_frac * ((1 - x_frac) * img[y + 1, x] + x_frac * img[y + 1, x + 1])
        return new_image
    img = resize_image(img, tiny_size)
    feature = img.flatten()
    feature = feature.reshape(-1, 1)
    feature = (feature - np.mean(feature)) / np.std(feature)
    return feature

def predict_knn(feature_train, label_train, feature_test, n_neighbors):
    num_test = feature_test.shape[0]
    pred_test = np.zeros(num_test, dtype=label_train.dtype)
    for i in range(num_test):
        distances = np.sum((feature_train - feature_test[i, :]) ** 2, axis=1)
        knn_indices = np.argsort(distances)[:n_neighbors]
        knn_labels = label_train[knn_indices]
        knn_labels = np.array(knn_labels).flatten()
        pred_test[i] = np.bincount(knn_labels).argmax()
    return pred_test

def compute_confusion_matrix_and_accuracy(pred, label, n_classes):
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)

    for i in range(len(pred)):
        confusion[label[i], pred[i]] += 1

    accuracy = np.trace(confusion) / np.sum(confusion)
    return confusion, accuracy

def compute_dsift(img, stride, size):
    sift = cv2.SIFT_create()
    kps = []
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            kp = cv2.KeyPoint(i, j, size)
            kps.append(kp)
    _, dsift = sift.compute(img, kps)
    return dsift

def build_visual_dictionary(features, dict_size):
    km = KMeans(n_clusters=dict_size, n_init=200, max_iter=20).fit(features)
    vocab = km.cluster_centers_
    return vocab

def compute_bow(dsift, vocab):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vocab)
    matches = neigh.kneighbors(dsift, n_neighbors=1, return_distance=False)

    bow_feature = np.zeros((1, vocab.shape[0]))
    for i in range(0, matches.shape[0]):
        t = matches[i][0]
        bow_feature[0][t] += 1

    bow_feature = bow_feature / matches.shape[0]

    bow_feature = bow_feature.T
    return bow_feature

def predict_svm(feature_train, label_train, feature_test, n_classes):
    clf = SVC(C=9.5)
    clf.fit(feature_train, label_train)
    pred_test = clf.predict(feature_test)
    pred_test = pred_test.astype(int)
    pred_test[pred_test < 0] = 0
    pred_test[pred_test >= n_classes] = n_classes - 1
    return pred_test

def extract_dataset_info(data_path):
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, image_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        image_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, image_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        image_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows
    return label_classes, label_train_list, image_train_list, label_test_list, image_test_list

def get_scene_classification_data(data_dir):
    label_classes, label_train_list, image_train_list, label_test_list, image_test_list = \
        extract_dataset_info(data_dir)

    image_train, label_train, image_test, label_test = [], [], [], []
    for i, img_path in enumerate(image_train_list):
        image_train.append(cv2.imread(img_path, 0))
        label_train.append(label_classes.index(label_train_list[i]))
    for i, img_path in enumerate(image_test_list):
        image_test.append(cv2.imread(img_path, 0))
        label_test.append(label_classes.index(label_test_list[i]))

    label_train = np.array(label_train).reshape((-1, 1))
    label_test = np.array(label_test).reshape((-1, 1))

    return image_train, label_train, image_test, label_test, label_classes


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Change work_dir if the data is put in a different directory
    work_dir = './'

    # Data preparation
    image_train, label_train, image_test, label_test, label_classes = get_scene_classification_data(
        work_dir + 'scene_classification_data')

    # Tiny + KNN
    feature_train = np.hstack([get_tiny_image(img, (16, 16)) for img in image_train]).T  # (1500, 256)
    feature_test = np.hstack([get_tiny_image(img, (16, 16)) for img in image_test]).T  # (1500, 256)
    n_neighbors = 5
    pred_test = predict_knn(feature_train, label_train, feature_test, n_neighbors)  # (1500, 1)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)


    # Bag-of-words + KNN / SVM

    # Bag-of-words
    # 1. extract dense sift features
    stride, keypoint_size = 16, 16
    dsift_train = [compute_dsift(image, stride, keypoint_size) for image in tqdm(
        image_train, 'Extracting dense SIFT features for images from train set')]  # a list of (n, 128)
    dsift_test = [compute_dsift(image, stride, keypoint_size) for image in tqdm(
        image_test, 'Extracting dense SIFT features for images from test set')]  # a list of (n, 128)

    # 2. build dictionary from train data
    dic_size = 50
    vocab = build_visual_dictionary(np.vstack(dsift_train), dic_size)

    # 3. extract bag-of-words features
    # print((np.hstack([compute_bow(dsift, vocab) for dsift in dsift_train]).T).shape)
    feature_train = np.hstack([compute_bow(dsift, vocab) for dsift in dsift_train]).T  # (n_train, dic_size)
    feature_test = np.hstack([compute_bow(dsift, vocab) for dsift in dsift_test]).T  # (n_test, dic_size)

    # KNN
    n_neighbors = 5
    pred_test = predict_knn(feature_train, label_train, feature_test, n_neighbors)  # (1500, 1)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    # SVM
    pred_test = predict_svm(feature_train, label_train, feature_test, len(label_classes))  # (1500, 1)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)