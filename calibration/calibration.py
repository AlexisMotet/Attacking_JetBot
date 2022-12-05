import numpy as np
import cv2 as cv
import glob

NROWS = 3
NCOLUMNS = 3

if __name__=="__main__" :
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NROWS*NCOLUMNS,3), np.float32)
    objp[:,:2] = np.mgrid[0:NROWS,0:NCOLUMNS].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('images_distorsion/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (NROWS,NCOLUMNS), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print('image %s is correct' % fname)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (NROWS,NCOLUMNS), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
        else :
            print('image %s not correct' % fname)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savetxt('camera_matrix.txt', mtx, fmt='%f')
    np.savetxt('distorsion_coefficients.txt', dist, fmt='%f')

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( 'total error: {}'.format(mean_error/len(objpoints)) )