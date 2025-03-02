import cv2 as cv
import numpy as np
from typing import List, Tuple

def find_inliers_count(img1: np.ndarray, img2: np.ndarray) -> int:
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype(np.float32)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype(np.float32)
    threshold = 0.05
    radius = 2

    A_harris = cv.cornerHarris(img1_gray, blockSize=2, ksize=3, k=0.04)
    B_harris = cv.cornerHarris(img2_gray, blockSize=2, ksize=3, k=0.04)
    A_corners = np.argwhere(A_harris > threshold * A_harris.max())
    B_corners = np.argwhere(B_harris > threshold * B_harris.max())
    A_keypoints = [cv.KeyPoint(float(c[1]), float(c[0]), radius*2) for c in A_corners]
    B_keypoints = [cv.KeyPoint(float(c[1]), float(c[0]), radius*2) for c in B_corners]

    sift = cv.SIFT_create()
    A_kp, A_des = sift.compute(img1_gray.astype(np.uint8), A_keypoints)
    B_kp, B_des = sift.compute(img2_gray.astype(np.uint8), B_keypoints)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.match(A_des, B_des)
    matches = sorted(matches, key=lambda x: x.distance)
    putative_matches = matches[:200]

    if len(putative_matches) >= 4:
        src_pts = np.float32([A_kp[m.queryIdx].pt for m in putative_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([B_kp[m.trainIdx].pt for m in putative_matches]).reshape(-1, 1, 2)

        _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 0.3, maxIters=5000)
        return np.sum(mask)
    return 0

def find_stitch_order(imgs: List[np.ndarray]) -> List[Tuple[int, int]]:
    n = len(imgs)

    inliers_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            inliers = find_inliers_count(imgs[i], imgs[j])
            inliers_matrix[i, j] = inliers
            inliers_matrix[j, i] = inliers

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, inliers_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    stitched = set()
    stitch_order = []

    for pair in pairs:
        i, j, _ = pair
        if i not in stitched or j not in stitched:
            stitch_order.append((i, j))
            stitched.add(i)
            stitched.add(j)

        if len(stitched) == n:
            break

    return stitch_order

def stitch_pair(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    A_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype(np.float32)
    B_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype(np.float32)

    # Harris corner
    threshold = 0.05
    radius = 2
    A_harris = cv.cornerHarris(A_gray, blockSize=2, ksize=3, k=0.04)
    B_harris = cv.cornerHarris(B_gray, blockSize=2, ksize=3, k=0.04)
    A_corners = np.argwhere(A_harris > threshold * A_harris.max())
    B_corners = np.argwhere(B_harris > threshold * B_harris.max())
    A_keypoints = [cv.KeyPoint(float(c[1]), float(c[0]), radius*2) for c in A_corners]
    B_keypoints = [cv.KeyPoint(float(c[1]), float(c[0]), radius*2) for c in B_corners]

    # SIFT
    sift = cv.SIFT_create()
    A_kp, A_des = sift.compute(A_gray.astype(np.uint8), A_keypoints)
    B_kp, B_des = sift.compute(B_gray.astype(np.uint8), B_keypoints)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.match(A_des, B_des)
    matches = sorted(matches, key=lambda x: x.distance)
    putative_matches = matches[:200]
    src_pts = np.float32([A_kp[m.queryIdx].pt for m in putative_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([B_kp[m.trainIdx].pt for m in putative_matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 0.3, maxIters=5000)
    height_A, width_A = img1.shape[:2]
    height_B, width_B = img2.shape[:2]

    corners_A = np.float32([[0, 0], [0, height_A], [width_A, height_A], [width_A, 0]]).reshape(-1, 1, 2)
    corners_B = np.float32([[0, 0], [0, height_B], [width_B, height_B], [width_B, 0]]).reshape(-1, 1, 2)

    transformed_corners_A = cv.perspectiveTransform(corners_A, H)
    all_corners = np.concatenate((transformed_corners_A, corners_B), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]],
                            [0, 1, translation_dist[1]],
                            [0, 0, 1]])
    panorama_size = (x_max - x_min, y_max - y_min)
    panorama = cv.warpPerspective(img1, H_translation.dot(H), panorama_size)

    panorama[translation_dist[1]:translation_dist[1]+height_B, translation_dist[0]:translation_dist[0]+width_B] = img2

    return panorama

def stitch_multi(img_paths: List[str]) -> np.ndarray:
    imgs = [cv.imread(path) for path in img_paths]
    stitch_order = find_stitch_order(imgs)
    panorama = imgs[stitch_order[0][0]]
    for pair in stitch_order:
        idx1, idx2 = pair
        img_to_add = imgs[idx2] if imgs[idx1] is panorama else imgs[idx1]
        panorama = stitch_pair(panorama, img_to_add)

    return panorama

def main():
    # hill, ledge, pier are the categories
    category = "hill"
    img_paths = [
        f'/Users/sanpandey/Computer Vision/Assignment3/data/{category}/1.JPG',
        f'/Users/sanpandey/Computer Vision/Assignment3/data/{category}/2.JPG',
        f'/Users/sanpandey/Computer Vision/Assignment3/data/{category}/3.JPG'
    ]
    panorama = stitch_multi(img_paths)
    cv.imwrite(f'/Users/sanpandey/Computer Vision/Assignment3/stitchMulti/{category}/{category}_pano.jpg', panorama)

if __name__ == '__main__':
    main()