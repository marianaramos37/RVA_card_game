import cv2
import numpy as np
from threading import Thread
  
class Webcam:
  
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
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
        with np.load('webcam_calibration_ouput.npz') as X:
            mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
  
        # set up criteria, object points and axis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
          
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
  
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
  
        # find grid corners in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
  
        if ret == True:
              
            # project 3D points to image plane
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
            imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
  
            # draw cube
            self._draw_cube(image, imgpts)
  
    def _draw_cube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
  
        # draw floor
        cv2.drawContours(img, [imgpts[:4]],-1,(200,150,10),-3)
  
        # draw pillars
        for i,j in zip(range(4),range(4,8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
  
        # draw roof
        cv2.drawContours(img, [imgpts[4:]],-1,(200,150,10),3)










# set up classes
webcam = Webcam()
webcam.start()
effects = Effects()
  
# loop forever
while True:
   
    # if lego policeman detected...
    image = webcam.get_current_frame()
   
    # ...then draw a virtual jail for all those nasty criminals
    effects.render(image)
                      
    # show the scene
    cv2.imshow('scene',image)
    cv2.waitKey(100)