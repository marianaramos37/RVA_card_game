import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])

point1 = np.float32([[0.5,0.5,-1.5]])
point2 = np.float32([[0.5,0.5,0]])

cube = np.float32([[0.2,0.2,-0.5], [0.8,0.2,-0.5], [0.8,0.8,-0.5], [0.2,0.8,-0.5],
                   [0.2,0.2,-1],[0.8,0.2,-1],[0.8,0.8,-1],[0.2,0.8,-1]])
                   
def draw_trophy(frame, corners, mtx, dist): # esta é a matriz dos parametros intrinsecos

    h, w = np.shape(frame)[:2]
    
    if(corners is not None and len(corners)==4):
        corners = np.array(corners, np.float32)

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors. Esta é a matriz parametros extrinsecos
        _, rvecs, tvecs, _ = cv.solvePnPRansac(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_P3P)

        # project 3D points to image plane
        imgpts_cube, _ = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)
        imgpts_cube = np.int32(imgpts_cube).reshape(-1,2)

        imgpts_point1, _ = cv.projectPoints(point1, rvecs, tvecs, mtx, dist)
        imgpts_point1 = np.int32(imgpts_point1).reshape(-1,2)

        imgpts_point2, _ = cv.projectPoints(point2, rvecs, tvecs, mtx, dist)
        imgpts_point2 = np.int32(imgpts_point2).reshape(-1,2)

        corners = np.array(corners, np.int32)

        cv.line(frame, tuple(imgpts_point2[0]), tuple(imgpts_cube[0]),(4,139,171),6)
        cv.line(frame, tuple(imgpts_point2[0]), tuple(imgpts_cube[1]),(4,139,171),6)
        cv.line(frame, tuple(imgpts_point2[0]), tuple(imgpts_cube[2]),(4,139,171),6)
        cv.line(frame, tuple(imgpts_point2[0]), tuple(imgpts_cube[3]),(4,139,171),6)

        # cube floor
        cv.drawContours(frame, [imgpts_cube[:4]],-1,(5,180,221),-3)
        cv.drawContours(frame, [imgpts_cube[:4]],-1,(4,139,171),3)

        # draw pillars
        for i,j in zip(range(4),range(4,8)):
            cv.line(frame, tuple(imgpts_cube[i]), tuple(imgpts_cube[j]),(4,139,171),6)

        # draw top
        cv.drawContours(frame, [imgpts_cube[4:]],-1,(5,180,221),-3)
        cv.drawContours(frame, [imgpts_cube[4:]],-1,(4,139,171),3)

        cv.line(frame, tuple(imgpts_point1[0]), tuple(imgpts_cube[4]),(4,139,171),6)
        cv.line(frame, tuple(imgpts_point1[0]), tuple(imgpts_cube[5]),(4,139,171),6)
        cv.line(frame, tuple(imgpts_point1[0]), tuple(imgpts_cube[6]),(4,139,171),6)
        cv.line(frame, tuple(imgpts_point1[0]), tuple(imgpts_cube[7]),(4,139,171),6)

       
        return