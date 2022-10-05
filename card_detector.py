import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim 

NUM_CARDS = 4

####### Compare captured contours with card of the database

def compare_contours(captured_image, cards_corners, cards_normal):

    # Get corners and dimensions of the card on the database
    reference_image = cv2.imread(cards_normal[0])
    height, width = reference_image.shape[:-1]

    # Initial original image and dst corners
    original = np.array(cards_corners, np.float32)
    dst = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], np.float32)

    # For every rotation (horizontal or vertical)
    for i in range(2):

        # Rotate real image
        if i == 1:
            original = np.array([cards_corners[1], cards_corners[2], cards_corners[3], cards_corners[0]], np.float32)

        # Compute homography
        matrix = cv2.getPerspectiveTransform(original, dst)

        # Generate frontal view from perspective
        card_frontal_view = cv2.warpPerspective(captured_image, matrix, (width, height))    

        # Compare images with cards on the database
        for card in cards_normal:
            card_of_the_database = binarize_image(cv2.imread(card))
            similarity = compute_similarity(card_frontal_view, card_of_the_database)
            if similarity < 15000:
                return card



####### Get the corners of a card through its countor 

def get_corners(contour):
    return cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)



####### Compare 2 images of the same size

def compute_similarity(image1, image2):  # Can we use any openCV function ?

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
    height, width = image1.shape
    errorL2 = cv2.norm( image1, image2, cv2.NORM_L2 )
    similarity = 1 - errorL2 / ( height * width )
    '''

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
    
    histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    i = 0
    c1=0
    while i<len(histogram1) and i<len(histogram2):
        c1+=(histogram1[i]-histogram2[i])**2
        i+= 1
    similarity = c1**(1 / 2)

    return similarity
    


####### Calibrate camera with chessboard images
####### https://learnopencv.com/camera-calibration-using-opencv/

def calibrate_camera(img):
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('images/chessboard_calibration/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        if ret:
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, 



####### Binarize an image

def binarize_image(original_image):
    gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_scale_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image



####### Do card recognition on just a test image - To be deleted once it is working on the main function

def test_on_image():
    cards_normal = glob.glob('./images/cards_normal/*')

    test_image = cv2.imread('./images/test.jpg')
    test_image = cv2.resize(test_image,(500,500))

    # Binarize 
    binary_image = binarize_image(test_image)

    contours, hierarchy = cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:NUM_CARDS]  

    cv2.drawContours(test_image, contours, -1, (255, 0, 0), 3)
 
    card_names = []

    for card_contour in contours:
        corners = get_corners(card_contour)
        x=0
        y=0
        for corner in corners:
            x,y = corner.ravel()
            cv2.circle(test_image, (x,y), 1, (0,0,255), cv2.LINE_AA)
            
        # Compute similarity and return name of each card
        original = compare_contours(binary_image, get_corners(card_contour), cards_normal)

        if original != None:
            cv2.putText(test_image,original[-14:],(x-150,y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        card_names.append(original)

    print(card_names)

    cv2.imshow('test', test_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



####### Video capture main loop

def main():
    cap = cv2.VideoCapture(0)
    cards_normal = glob.glob('images/cards_normal/*.png')

    while True:
        _, frame = cap.read()

        # Calibrate camera
        # _, cameraMatrix, distortionCoefficients = calibrate_camera(frame)

        # Binarize 
        binary_frame = binarize_image(frame)

        contours, hierarchy = cv2.findContours(binary_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)[:NUM_CARDS]  

        cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

        for card in contours:
            corners = get_corners(card)
            for corner in corners:
                x,y = corner.ravel()
                cv2.circle(frame, (x,y), 1, (0,0,255), cv2.LINE_AA)

        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

test_on_image()
#main()