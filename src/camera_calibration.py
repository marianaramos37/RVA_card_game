import numpy as np
import cv2 as cv
import glob


def calibrate_camera():

    chessboardSize = (9,6)
    frameSize = (640,360)

    # Prepare actual object chessboard points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    # Load chessboard images
    images = glob.glob('images/chessboard_calibration/*.jpg') 

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image in images:

        img = cv.imread(image)
        h, w = np.shape(img)[:2]

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            cv.rectangle(img, (10,10), (w-15, 85), (255,255,255), -1)
            cv.rectangle(img, (10,10), (w-15, 85), (0, 0, 0), 2)
            cv.putText(img,"Calibrating the camera...",(20,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.FONT_HERSHEY_SIMPLEX)
            cv.putText(img,"This might take a minute.",(20,65), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.FONT_HERSHEY_SIMPLEX)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(70)

    cv.destroyAllWindows()

    # Calibrate the camera
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    return ret, cameraMatrix, dist, rvecs, tvecs
