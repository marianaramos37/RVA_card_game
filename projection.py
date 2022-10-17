import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1*4,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]]).reshape(-1,3) # (x,y,z)

cube = np.float32([[0,0,0], [0,0.5,0], [0.5,0.5,0], [0.5,0,0],
                   [0,0,-0.5],[0,0.5,-0.5],[0.5,0.5,-0.5],[0.5,0,-0.5] ])
                   
def draw_trophy(frame, corners, mtx, dist):

    # mtx = [[focal_length, 0, center[0]],
    #        [0, focal_length, center[1]],
    #        [0, 0, 1]]
    
    corners = np.array(corners, np.float32)

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    if(len(corners)>0):
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs,_ = cv.solvePnPRansac(objectPoints=objp, imagePoints=corners2, cameraMatrix=mtx, distCoeffs=dist)

        # project 3D points to image plane
        imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        imgpts = np.int32(imgpts).reshape(-1,2)

        corners = np.array(corners, np.int32)

        cv.circle(frame,tuple(corners[0].ravel()),1,[0,255,255],2)
        cv.circle(frame,tuple(corners[1].ravel()),1,[0,255,255],2)
        cv.circle(frame,tuple(corners[2].ravel()),1,[0,255,255],2)
        cv.circle(frame,tuple(corners[3].ravel()),1,[0,255,255],2)

        corner = tuple(corners[0].ravel())
        cv.line(frame, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
        cv.line(frame, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
        cv.line(frame, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)

        return
