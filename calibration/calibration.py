
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

# Import required modules
import cv2
import numpy as np
import os

"""
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
What about the 3D points from real world space? Those images are 
taken from a static camera and chess boards are placed at different 
locations and orientations. So we need to know (X,Y,Z) values. But 
for simplicity, we can say chess board was kept stationary at XY 
plane, (so Z=0 always) and camera was moved accordingly. This 
consideration helps us to find only X,Y values. Now for X,Y values, 
we can simply pass the points as (0,0), (1,0), (2,0), ... which 
denotes the location of points. In this case, the results we get 
will be in the scale of size of chess board square. But if we know 
the square size, (say 30 mm), we can pass the values as (0,0), 
(30,0), (60,0), ... . Thus, we get the results in mm. (In this case, 
we don't know square size since we didn't take those images, so we 
pass in terms of square size).

This function may not be able to find the required pattern in all 
the images. So, one good option is to write the code such that, it 
starts the camera and check each frame for required pattern. Once 
the pattern is obtained, find the corners and store it in a list. 
Also, provide some interval before reading next frame so that we can 
adjust our chess board in different direction. Continue this process 
until the required number of good patterns are obtained. Even in the 
example provided here, we are not sure how many images out of the 14 
given are good. Thus, we must read all the images and take only the 
good ones.
Instead of chess board, we can alternatively use a circular grid. In 
this case, we must use the function cv.findCirclesGrid() to find the 
pattern. Fewer images are sufficient to perform camera calibration 
using a circular grid.
"""
show = False

def reprojection_error(points_2d, points_3d, mtx, dist, rvecs, tvecs, fisheye):
    mean_error = 0
    if fisheye :
        points_3d = np.expand_dims(np.asarray(points_3d), -2)
    for i in range(len(points_3d)):
        if fisheye :
            points_2d_projected, _ = cv2.fisheye.projectPoints(points_3d[i], 
                                                   rvecs[i], 
                                                   tvecs[i], 
                                                   mtx, dist)
        else : 
            points_2d_projected, _ = cv2.projectPoints(points_3d[i], 
                                                   rvecs[i], 
                                                   tvecs[i], 
                                                   mtx, dist)
        error = cv2.norm(points_2d[i], points_2d_projected, 
                         cv2.NORM_L2)/len(points_2d_projected)
        mean_error += error
    print("total error: %f" % (mean_error/len(points_3d)))
    return mean_error/len(points_3d)
    

def calibrate(path, chessboard, show, fisheye):
    # Define the dimensions of chessboard
    assert type(chessboard) == tuple
    CHESSBOARD = chessboard
    
    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    
    # Vector for 3D points
    points_3d = []
    
    # Vector for 2D points
    points_2d = []
    
    # 3D points real world coordinates
    point_3d = np.zeros((CHESSBOARD[0] * CHESSBOARD[1],3), 
                          np.float32)
    point_3d[:, :2] = np.mgrid[0:CHESSBOARD[0],
                                   0:CHESSBOARD[1]].T.reshape(-1, 2)

    # Extracting path of individual image stored
    # in a given directory. Since no path is
    # specified, it will take current directory
    # jpg files alone
    images = os.listdir(path)
    for filename in images:
        image = cv2.imread(path + filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        gray, CHESSBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK 
                        + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of chess board
        if not ret :
            continue
        points_3d.append(point_3d)

        # Refining pixel coordinates
        # for given 2d points.
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        points_2d.append(corners)

        if show : 
            # Draw and display the corners
            image = cv2.drawChessboardCorners(image,
                                            CHESSBOARD,
                                            corners, ret)
            cv2.imshow('img', image)
            cv2.waitKey(0)
    if show :
        cv2.destroyAllWindows()
    if len(points_3d) == 0 :
        return None
    if fisheye :
        points_3d_ = np.expand_dims(np.asarray(points_3d), -2)
        _, matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            points_3d_, points_2d, gray.shape[::-1], None, None)
    else :
        _, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            points_3d, points_2d, gray.shape[::-1], None, None)
    
    return points_2d, points_3d, matrix, distortion, rvecs, tvecs  
 
def undistort(mtx, dist, path_img, fisheye):
    img = cv2.imread(path_img)
    h, w = img.shape[:2]
    if fisheye :
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, None, mtx, (h, w), cv2.CV_16SC2)
    else :
        map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (h, w), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def look_for_best(path):
    min_error = 1e3
    best_matrix, best_distortion = None, None
    for i in range(3, 10):
        for j in range(3, 10):
            res = calibrate(path, (i, j), show=False, fisheye=False)
            if res is not None :
                points_2d, points_3d, matrix, distortion, rvecs, tvecs = res
                error = reprojection_error(points_2d, points_3d, matrix, distortion, rvecs, tvecs, fisheye=False)
                if (error is not None and error < min_error) :
                    best_matrix, best_distortion, min_error = matrix, distortion, error
    print("min error : %f" % min_error)
    return  best_matrix, best_distortion
    
            
if __name__=="__main__" :
    path = "calibration\\images_calibration\\"
    """
    points_2d, points_3d, matrix, distortion, rvecs, tvecs = calibrate(path, (6, 6), show=False, fisheye=False)
    reprojection_error(points_2d, points_3d, matrix, distortion, 
                       rvecs, tvecs, fisheye=False)
    path_img = "C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\images_distortion\\6f6f2002-682a-11ed-a72e-4a433b536649.jpg"
    undistort(matrix, distortion, path_img, fisheye=False)
    """
    """
    mtx, dist = look_for_best(path)
    np.savetxt('camera_matrix.txt', mtx, fmt='%f')
    np.savetxt('distortion_coefficients.txt', dist, fmt='%f')
    """
    res = calibrate(path, (6, 9), show=False, fisheye=False)
    if res is not None :
        points_2d, points_3d, matrix, distortion, rvecs, tvecs = res
        error = reprojection_error(points_2d, points_3d, matrix, distortion, rvecs, tvecs, fisheye=False)
        print(error)
        print(distortion)