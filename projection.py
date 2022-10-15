import numpy as np
import cv2 as cv
import glob

from camera_calibration import calibrate_camera 

_, mtx, dist, _, _ = calibrate_camera()

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-10],[0,3,-10],[3,3,-10],[3,0,-10] ])
                   
def draw_trophy(frame, corners):
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(frame, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        frame = cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
       
    # draw top layer in red color
    frame = cv.drawContours(frame, [imgpts[4:]],-1,(0,0,255),3)

    return img
