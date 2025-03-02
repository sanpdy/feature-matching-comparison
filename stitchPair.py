import cv2 as cv
import numpy as np

def draw_features(img, keypoints):
    img_copy = img.copy()
    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv.circle(img_copy, (x, y), 3, (0, 255, 255), -1)
    return img_copy

def draw_matches(img1, kp1, img2, kp2, matches, mask=None, color=(255, 0, 0)):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    output[:h1, :w1] = img1
    output[:h2, w1:w1+w2] = img2

    for idx, match in enumerate(matches):
        if mask is None or mask[idx]:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, (kp2[match.trainIdx].pt[0] + w1, kp2[match.trainIdx].pt[1])))
            cv.line(output, pt1, pt2, color, 1)

    return output

def stitch_pair(img_src_path: str, img_des_path: str) -> np.ndarray:
    # Load images and convert to grayscale and np.float32
    A = cv.imread(img_src_path)
    B = cv.imread(img_des_path)
    A_gray = cv.cvtColor(A, cv.COLOR_BGR2GRAY).astype(np.float32)
    B_gray = cv.cvtColor(B, cv.COLOR_BGR2GRAY).astype(np.float32)

    # Feature points using Harris corner detector
    threshold = 0.05
    radius = 2
    A_harris = cv.cornerHarris(A_gray, blockSize=2, ksize=3, k=0.04)
    B_harris = cv.cornerHarris(B_gray, blockSize=2, ksize=3, k=0.04)

    # Threshold
    A_corners = np.argwhere(A_harris > threshold * A_harris.max())
    B_corners = np.argwhere(B_harris > threshold * B_harris.max())

    # Corners to KeyPoint objects
    A_keypoints = [cv.KeyPoint(float(c[1]), float(c[0]), radius*2) for c in A_corners]
    B_keypoints = [cv.KeyPoint(float(c[1]), float(c[0]), radius*2) for c in B_corners]
    A_features = draw_features(A, A_keypoints)
    B_features = draw_features(B, B_keypoints)
    cv.imwrite('/Users/sanpandey/Computer Vision/Assignment3/stitchPair/features_A.jpg', A_features)
    cv.imwrite('/Users/sanpandey/Computer Vision/Assignment3/stitchPair/features_B.jpg', B_features)

    # SIFT
    sift = cv.SIFT_create()
    A_kp, A_des = sift.compute(A_gray.astype(np.uint8), A_keypoints)
    B_kp, B_des = sift.compute(B_gray.astype(np.uint8), B_keypoints)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.match(A_des, B_des)

    # Putative matches
    matches = sorted(matches, key=lambda x: x.distance)
    putative_matches = matches[:200]
    putative_viz = draw_matches(A, A_kp, B, B_kp, putative_matches, color=(255, 0, 0))
    cv.imwrite('/Users/sanpandey/Computer Vision/Assignment3/stitchPair/putative_matches.jpg', putative_viz)

    # RANSAC
    src_pts = np.float32([A_kp[m.queryIdx].pt for m in putative_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([B_kp[m.trainIdx].pt for m in putative_matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 0.3, maxIters=5000)

    # Inlier matches
    inlier_viz = draw_matches(A, A_kp, B, B_kp, putative_matches, mask.ravel().tolist(), (0, 255, 0))
    cv.imwrite('/Users/sanpandey/Computer Vision/Assignment3/stitchPair/inlier_matches.jpg', inlier_viz)

    # Pano
    height_A, width_A = A.shape[:2]
    height_B, width_B = B.shape[:2]

    corners_A = np.float32([[0, 0], [0, height_A], [width_A, height_A], [width_A, 0]]).reshape(-1, 1, 2)
    corners_B = np.float32([[0, 0], [0, height_B], [width_B, height_B], [width_B, 0]]).reshape(-1, 1, 2)

    transformed_corners_A = cv.perspectiveTransform(corners_A, H)
    all_corners = np.concatenate((transformed_corners_A, corners_B), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]],[0, 1, translation_dist[1]],[0, 0, 1]])
    panorama_size = (x_max - x_min, y_max - y_min)
    panorama = cv.warpPerspective(A, H_translation.dot(H), panorama_size)
    panorama[translation_dist[1]:translation_dist[1]+height_B, translation_dist[0]:translation_dist[0]+width_B] = B
    return panorama

def main():
    A = '/Users/sanpandey/Computer Vision/Assignment3/data/uttower_left.jpg'
    B = '/Users/sanpandey/Computer Vision/Assignment3/data/uttower_right.jpg'
    panorama = stitch_pair(A, B)
    cv.imwrite('/Users/sanpandey/Computer Vision/Assignment3/stitchPair/stitchPairPano.jpg', panorama)

if __name__ == '__main__':
    main()