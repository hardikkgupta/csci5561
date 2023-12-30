import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import interpn



def find_match(img1, img2):
    # img1.shape           # (234, 291)
    # img2.shape           # (1008, 1344)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    # len(kp1)              # 576
    # des1.shape            # (576, 128)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # len(kp2)              # 2366
    # des1.shape            # (2366, 128)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)
    distances, indices = nbrs.kneighbors(des1)
    # distances.shape       # (576, 2)
    # For each keypoint, it is finding the distance from the 2 neighbours nearest in target image
    # indices.shape         # (576, 2)
    # For each keypoint, it is finding the indices from the 2 neighbours nearest in target image
    x1 = []
    x2 = []
    for i in range(len(distances)):
        if (distances[i, 0] / distances[i, 1]) < 0.68:
            x1.append(kp1[i].pt)
            x2.append(kp2[indices[i, 0]].pt)
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    best_inliers = []
    # A, mask = cv2.estimateAffine2D(x1, x2, ransacReprojThreshold=ransac_thr)
    # new_row = np.array([0, 0, 1])
    # A = np.vstack((A, new_row))
    # A = A.reshape(3,3)
    # print(A)

    for _ in range(ransac_iter):
        sample_indices = np.random.choice(len(x1), size=3, replace=False)
        sample_x1 = x1[sample_indices]
        sample_x2 = x2[sample_indices]

        x1_h = np.hstack((sample_x1, np.ones((3, 1))))
        x2_h = np.hstack((sample_x2, np.ones((3, 1))))
        A, _, _, _ = np.linalg.lstsq(x1_h, x2_h, rcond=None)

        transformed_x1_h = np.hstack((x1, np.ones((len(x1), 1))))
        transformed_x1 = np.dot(transformed_x1_h, A.T)[:, :2]

        residuals = np.linalg.norm(transformed_x1 - x2, axis=1)
        inliers = np.where(residuals < ransac_thr)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    best_x1 = x1[best_inliers]
    best_x2 = x2[best_inliers]
    x1_h = np.hstack((best_x1, np.ones((len(best_x1), 1))))
    x2_h = np.hstack((best_x2, np.ones((len(best_x2), 1))))
    A, _, _, _ = np.linalg.lstsq(x1_h, x2_h, rcond=None)
    A = A.T

    return A


def warp_image(img, A, output_size):
    input_shape = img.shape[:2]
    x_output, y_output = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))
    transformed_coords = np.dot(A, np.vstack([x_output.flatten(), y_output.flatten(), np.ones_like(x_output.flatten())]))
    transformed_coords[0, :] = np.clip(transformed_coords[0, :], 0, input_shape[1] - 1)
    transformed_coords[1, :] = np.clip(transformed_coords[1, :], 0, input_shape[0] - 1)
    x_transformed = transformed_coords[0, :].reshape(output_size)
    y_transformed = transformed_coords[1, :].reshape(output_size)
    img_warped = interpn((np.arange(input_shape[0]), np.arange(input_shape[1])),
                         img, (y_transformed, x_transformed), method='linear', bounds_error=False, fill_value=0)
    return img_warped

def visualize_sift(img):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sqrt(np.sum(np.square(x2_t[:, :2] - x2), axis=1))
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y', linewidth=3)
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_warp_image(img_warped, img):
    plt.subplot(131)
    plt.imshow(img_warped, cmap='gray')
    plt.title('Warped image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.abs(img_warped - img), cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./template.jpg', 0)  # read as grey scale image
    target = cv2.imread('./target.jpg', 0)  # read as grey scale image

    x1, x2 = find_match(template, target)
    visualize_find_match(template, target, x1, x2)

    ransac_thr = 30  # specify error threshold for RANSAC (unit: pixel)
    ransac_iter = 5000  # specify number of iterations for RANSAC
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target, x1, x2, A, ransac_thr)

    img_warped = warp_image(target, A, template.shape)
    # plt.imshow(img_warped, cmap='gray')
    # plt.show()
    visualize_warp_image(img_warped, template)