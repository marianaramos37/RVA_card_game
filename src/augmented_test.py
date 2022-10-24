import cv2
import numpy as np
from threading import Thread
from camera_calibration import calibrate_camera
from projection import draw_trophy

####### Compare 2 images of the same size

def compute_similarity(image1, image2):

    # Template matching:
    '''
    similarity = cv2.matchTemplate(image1,image2,cv2.TM_CCOEFF)
    '''

    # Feature matching:
    '''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(image1,image2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    similarity = len(good) / len(matches)
    '''

    # Using norm():
    '''
    height1, width1 = image1.shape
    errorL2 = cv2.norm( image1, image2, cv2.NORM_L2 ) 
    similarity = 1 - errorL2 / ( height1 * width1 )
    '''

    # Using abs diff
    
    diff_img = cv2.absdiff(image1, image2)
    diff = int(np.sum(diff_img)/255)
    
    # By pixel, too slow:
    '''
    rest = cv2.bitwise_xor(image1, image2)
    height, width = image1.shape
    pixels = height * width
    diffs = 0
    for x in range(width):
        for y in range(height):
            if rest[y, x] != 0:
                diffs += 1

    similarity = (pixels - diffs) / float(pixels)
    '''

    # Using SSIM
    '''
    similarity = ssim(image1,image2, channel_axis = False)  
    '''

    # Histogram
    '''
    histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    i = 0
    c1=0
    while i<len(histogram1) and i<len(histogram2):
        c1+=(histogram1[i]-histogram2[i])**2
        i+= 1
    similarity = c1**(1 / 2)
    '''
    return diff
    
####### Get the corners of a card through its countor 

def get_corners(contour):
    return cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)
    
####### Binarize an image

def binarize_image(original_image):
    gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img_w, img_h = np.shape(original_image)[:2]

    bkg_level = gray_scale_image[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + 100

    _, binary_image = cv2.threshold(gray_scale_image, thresh_level, 255, cv2.THRESH_BINARY)
    return binary_image

#### Find the marker

def find_marker(thresh_image, template):
    height, width = template.shape[:2]
    cnts,_ = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        peri = cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
        if(len(approx)) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if  (0.5<=(float(w)/h)<=1.5):
                corners = get_corners(cnts[i])

                original = np.array(corners, np.float32)
                dst = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], np.float32)

                if(len(original)==4):
                    matrix = cv2.getPerspectiveTransform(original, dst)
                else: return

                marker_frontal_view = cv2.warpPerspective(thresh_image, matrix, (width, height)) # TODO: Falta rodar

                similarity = compute_similarity(marker_frontal_view, template)
                if similarity >= 51000:
                    return corners

    return []
    
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


 # load calibration data
_, mtx, dist, _, _ = calibrate_camera()

class Effects(object):
    
    def render(self, image, winner):
  
        # set up criteria, object points and axis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
          
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
  
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-6],[0,3,-6],[3,3,-6],[3,0,-6] ])
  
        # find grid corners in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        print(ret)
        if ret == True:
              
            # project 3D points to image plane
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            ret , rvecs, tvecs = cv2.solvePnPRansac(objp, corners, mtx, dist)[:3]

            imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
  
            if (winner == 'team1'):
                img_winner = cv2.imread('images/Team1Winner.png')
            else:
                img_winner = cv2.imread('images/Team2Winner.png')
                  
            # draw cube
            imgage3,image2=self._draw_cube(image, imgpts, img_winner)
            return imgage3,image2
  
    def _draw_cube(self, img, imgpts, img_winner):
        imgpts = np.int32(imgpts).reshape(-1,2)
  
        # draw floor
        cv2.drawContours(img, [imgpts[:4]],-1,(200,150,10),-3)
  
        # draw pillars
        for i,j in zip(range(4),range(4,8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        #7 4
        #3 0
        
        x = img_winner.shape[0]
        y = img_winner.shape[1]

        pts1=np.array([[x,y],[0,y],[x,0],[0,0]])
        pts2=np.array([imgpts[4], imgpts[7], [imgpts[4][0]+(imgpts[4][0]-imgpts[0][0]), imgpts[4][1]-(imgpts[0][1]-imgpts[4][1])], [imgpts[7][0]+(imgpts[7][0]-imgpts[3][0]), imgpts[7][1]-(imgpts[3][1]-imgpts[7][1])]])
        print(pts1)
        
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0)
        im1Reg = cv2.warpPerspective(img_winner, h, (640, 360))
        print(img.shape)
        
        # draw roof
        img=cv2.drawContours(img, [imgpts[4:]],-1,(200,150,10),-3)
        return img,im1Reg

    def _merge_images(self,image1,image2):
        # Load two images
        img1 = image1
        img2 = image2
        # I want to put logo on top-left corner, So I create a ROI
        rows,cols,_ = img2.shape
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

# set up class
effects = Effects()


# loop for every image

   

frame = cv2.imread('images/test_draw_trophy.png')
#draw cube
# # Pre-process marker image:
marker = cv2.imread('images/marker.png')
gray_scale_marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
_, marker = cv2.threshold(gray_scale_marker, 100, 255, cv2.THRESH_BINARY)
# Binarize frame
binary_frame = binarize_image(frame)

# Find and recognize marker
marker_corners = find_marker(binary_frame,marker)
# cap = cv2.VideoCapture(0)
# while True:
# img3,img2=effects.render(image, 'team1')
# res=effects._merge_images(img3,img2)
draw_trophy(frame, marker_corners, mtx, dist, 'teamchef')
cv2.imshow('Sueca Game Assistant', frame)
k = cv2.waitKey(0)
                    
# show the scene
cv2.waitKey(100)