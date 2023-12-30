# Stereo Reconstruction

## Overview
To understand better, let’s define the terminologies.
- Stereo: Refers to the use of two or more images taken from slightly different viewpoints, similar to how our eyes perceive depth. It is derived from the word from stereoscopy, refers to the technique of creating or enhancing the illusion of depth in an image by presenting two offset images separately to the left and right eyes of the observer.
- Geometric Aspects: Refers to the study of the spatial relationships and properties of objects. In stereo vision, it focuses on the geometric principles governing the formation of images from different viewpoints. Key geometric aspects include triangulation, epipolar geometry, and disparity mapping (each can defined in its own paragraphs).
- SIFT algorithm: SIFT is an algorithm used in computer vision for detecting and describing distinctive features in images, irrespective of their scale, orientation, and illumination changes. In stereo reconstruction, SIFT can be employed to find corresponding points between the left and right images, helping to establish reliable correspondences used in the geometric analysis for depth estimation.

## Fundamental Matrix Computation
Given matches, you will compute a fundamental matrix to draw epipolar lines.
```
 def compute_F(pts1, pts2):
    ...
    return F
```
**Input**: $pts1$ and $pts2$ are $n \times 2$ matrices that specify the correspondence. <br>
**Output**: $F \in R^{3×3}$ is the fundamental matrix.

- The **fundamental matrix** represents the geometric relationship between two images of a scene taken from different viewpoints.
- **SVD** is a mathematical technique used to decompose a matrix into three simpler matrices: $A = U \sum V T$ . In the context of computer vision and the 8-point algorithm, SVD is often used to solve systems of linear equations and estimate the fundamental matrix.
- **RANSAC** works by randomly sampling a minimal subset of the data, fitting the model to that subset, and then checking how well the model fits the rest of the data. This process is iterated to find the best model while ignoring outliers.
- The **8-point algorithm** is a method used in computer vision and image processing for estimating the fundamental matrix between two views of a scene. It requires at least 8 corresponding points in the two images. The algorithm uses these point correspondences to set up a linear system, and the solution is obtained through Singular Value Decomposition (SVD).

The fundamental matrix is defined by the equation: <br>
$xTFx = 0$ <br>
for any pair of matching points $x ⇿ x’$ in two images. In particular, writing $x = (x, y, 1)$ and $x’ = (x′, y′, 1)$. Given sufficiently many points,
$Af = (x′x,x′y,x′,y′x,y′y,y′,x,y,1)f = 0$
As written in Chapter 11, The least-squares solution for f is the singular vector corresponding to the smallest singular value of $A$, i.e. the last column of $V$ in SVD. Rank - 2 is enforced by setting the smallest singular value to 0.
`compute_camera_pose` computes poses (positions and orientations) in 3D space. The intrinsic matrix K includes information about the camera’s focal length, principal point.


<img width="1069" alt="Screenshot 2023-12-30 at 3 07 26 AM" src="https://github.com/hardikkgupta/csci5561/assets/40640596/f259dff2-37bc-465b-84e5-de5c9ef32857">
