import numpy as np
import cv2 as cv

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])

point1 = np.float32([[0.5,0.5,-1.5]])
point2 = np.float32([[0.5,0.5,0]])

cube = np.float32([[0.2,0.2,-0.5], [0.8,0.2,-0.5], [0.8,0.8,-0.5], [0.2,0.8,-0.5],
                   [0.2,0.2,-1],[0.8,0.2,-1],[0.8,0.8,-1],[0.2,0.8,-1]])
                   
def draw_trophy(frame, corners, mtx, dist, winning_team): # esta é a matriz dos parametros intrinsecos

    h, w = np.shape(frame)[:2]
    
    if(corners is not None and len(corners)==4):
        img_winner = cv.imread('images/trunfos.png')
        if (winning_team == 'team1'):
            img_winner = cv.imread('images/Team1Winner.png')
        elif (winning_team == 'team2'):
            img_winner = cv.imread('images/Team2Winner.png')
        
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
        
        #draw trophy image above cube
        x = img_winner.shape[0]
        y = img_winner.shape[1]

        pts1=np.array([[x,y],[0,y],[x,0],[0,0]])
        pts2=np.array([imgpts_cube[4], imgpts_cube[7], [imgpts_cube[4][0]+(imgpts_cube[4][0]-imgpts_cube[0][0]), imgpts_cube[4][1]-(imgpts_cube[0][1]-imgpts_cube[4][1])], [imgpts_cube[7][0]+(imgpts_cube[7][0]-imgpts_cube[3][0]), imgpts_cube[7][1]-(imgpts_cube[3][1]-imgpts_cube[7][1])]])
        
        homography, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 2.0)
        img_winner_warped = cv.warpPerspective(img_winner, homography, (640, 360))
        
        # I want to put logo on top-left corner, So I create a ROI
        rows,cols,_ = img_winner_warped.shape
        roi = frame[0:rows, 0:cols]
        
        # Now create a mask of logo and create its inverse mask also
        img_winner_warpedgray = cv.cvtColor(img_winner_warped, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(img_winner_warpedgray, 10, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        
        # Now black-out the area of logo in ROI
        frame_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
        
        # Take only region of logo from logo image.
        img_winner_warped_fg = cv.bitwise_and(img_winner_warped, img_winner_warped, mask = mask)
        
        # Put logo in ROI and modify the main image
        dst = cv.add(frame_bg, img_winner_warped_fg)
        frame[0:rows, 0:cols ] = dst
        return