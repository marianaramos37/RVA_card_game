import cv2
import numpy as np
import glob
from threading import Thread


from camera_calibration import calibrate_camera 



  
class Webcam:
  
    def __init__(self):
        self.video_capture = cv2.VideoCapture(1)
        self.current_frame = self.video_capture.read()[1]
          
    # create thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()
  
    def _update_frame(self):
        while(True):
            self.current_frame = self.video_capture.read()[1]
                  
    # get the current frame
    def get_current_frame(self):
        return self.current_frame



class Effects(object):
    
    def render(self, image):
  
        # load calibration data
        _, mtx, dist, _, _ = calibrate_camera()
  
        # set up criteria, object points and axis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
          
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
  
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-6],[0,3,-6],[3,3,-6],[3,0,-6] ])
  
        # find grid corners in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
  
        if ret == True:
              
            # project 3D points to image plane
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            ret , rvecs, tvecs = cv2.solvePnPRansac(objp, corners, mtx, dist)[:3]

            imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
  
            # draw cube
            imgage3,image2=self._draw_cube(image, imgpts)
            return imgage3,image2
  
    def _draw_cube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
  
        # draw floor
        cv2.drawContours(img, [imgpts[:4]],-1,(200,150,10),-3)
  
        # draw pillars
        for i,j in zip(range(4),range(4,8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        #0 4
        #3 7

        pts1=np.array([[403,390],[0,390],[403,0],[0,0]])
        pts2=np.array([[imgpts[0]],[imgpts[3]],[imgpts[4]],[imgpts[7]]])
        image2=cv2.imread('images/trunfos.png')
        
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,2.0)
        im1Reg = cv2.warpPerspective(image2, h, (640, 360))
        print(img.shape)
        
        # draw roof
        imgage3=cv2.drawContours(img, [imgpts[4:]],-1,(200,150,10),-3)
        return imgage3,im1Reg

    def _merge_images(self,image1,image2):
        # Load two images
        img1 = image1
        img2 = image2
        # I want to put logo on top-left corner, So I create a ROI
        rows,cols,channels = img2.shape
        roi = img1[0:rows, 0:cols]
        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        img1[0:rows, 0:cols ] = dst
        cv2.imshow('res',img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()










# set up classe
effects = Effects()


# loop for every image

   

#draw cube
image=cv2.imread('images/chessboard_calibration/cali16.jpg')
img3,img2=effects.render(image)
res=effects._merge_images(img3,img2)
cv2.imshow('mat1',res)
cv2.imshow('mat2',img2)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('badabada.png', img)
                    
# show the scene
cv2.waitKey(100)