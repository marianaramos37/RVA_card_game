import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
import sys

from camera_calibration import calibrate_camera 
import game_logic


NUM_CARDS = 4
CARD_MAX_AREA = 18000
CARD_MIN_AREA = 6000

####### Compare captured contours with card of the database

def compare_contours(captured_image, cards_corners, cards_db_names, cards_db):

    # Get corners and dimensions of the card on the database
    height, width = cards_db[0].shape[:2]

    # Initial original image and dst corners
    original = np.array(cards_corners, np.float32)
    dst = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], np.float32)

    less_difference = 230
    best_match = ""

    # For every rotation (horizontal or vertical)
    for i in range(4):

        # Rotate real image
        if i == 1:
            original = np.array([cards_corners[1], cards_corners[2], cards_corners[3], cards_corners[0]], np.float32)
        if i == 2:
            original = np.array([cards_corners[2], cards_corners[3], cards_corners[0], cards_corners[1]], np.float32)
        if i == 3:
            original = np.array([cards_corners[3], cards_corners[0], cards_corners[1], cards_corners[2]], np.float32)

        # Compute homography
        matrix = cv2.getPerspectiveTransform(original, dst)

        # Generate frontal view from perspective
        card_frontal_view = cv2.warpPerspective(captured_image, matrix, (width, height))    

        # Compare images with cards on the database
        for c in range(len(cards_db_names)):
            similarity = compute_similarity(card_frontal_view, cards_db[c])
            if similarity < less_difference:
                less_difference = similarity
                best_match = cards_db_names[c]
        
    return best_match

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
    height1, width1 = image1.shape
    errorL2 = cv2.norm( image1, image2, cv2.NORM_L2 ) 
    similarity = 1 - errorL2 / ( height1 * width1 )
    '''

    # Using abs diff
    
    diff_img = cv2.absdiff(image1, image2)
    similarity = int(np.sum(diff_img)/255)
    
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
    return similarity


####### Binarize an image

def binarize_image(original_image):
    gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img_w, img_h = np.shape(original_image)[:2]

    bkg_level = gray_scale_image[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + 130 

    _, binary_image = cv2.threshold(gray_scale_image, thresh_level, 255, cv2.THRESH_BINARY)
    return binary_image

# Finds all card-sized contours in a thresholded camera image. Returns the number of cards, and a list of card contours sorted from largest to smallest.

def find_cards(thresh_image):

    # Find contours and sort their indices by contour size
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    if len(cnts) == 0:
        return [], []
    
    cnts_sort = [] # List of sorted contours
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int) # [0,1,0,1,1,...] where 1 is card 

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card



####### Video capture main loop

def main():

    # Get trunfo
    trunfo = ""
    args = sys.argv[1:]
    if(len(args) != 0 and (args[0]=="copas" or args[0]=="paus" or args[0]=="espadas" or args[0]=="ouros")):
        trunfo = args[0]
        print("Starting the SUECA game. The trunfo resgistered is " + trunfo + ". Loading the cards... this might take a minute.")
    else:
        print("Please specify a correct value for the 'trunfo' card.")
        print("USAGE: card_detector.py [TRUNFO]")
        print("Where  [TRUNFO] can be one of the following: 'copas', 'paus', 'ouros', 'espadas'")
        return 

    cap = cv2.VideoCapture(1)

    # Pre-process database cards:

    cards_normal = glob.glob('images/cards_simple/*.jpg')
    cards_db_preprocessed = []

    for card_name in cards_normal:
        image = cv2.imread(card_name)
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_scale_image, 127, 255, cv2.THRESH_BINARY)
        cards_db_preprocessed.append(binary_image)

    # Calibrate camera:
    # ret, cameraMatrix = calibrate_camera()
    
    # Game Logic:

    assistir = ""
    card_team1_player1 = ""
    card_team1_player2 = ""
    card_team2_player1 = ""
    card_team2_player2 = ""
    number_of_cards_on_table = 0

    while True:
        _, frame = cap.read()

        # Binarize 
        binary_frame = binarize_image(frame)
        
        cnts_sort, cnt_is_card = find_cards(binary_frame)
        
        num_cards_on_table = np.count_nonzero(cnt_is_card==1)
        
        cv2.putText(frame,"Trunfo: " + trunfo,(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame,"Team 1: " + trunfo,(5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame,"Team 2: " + trunfo,(5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        for c in range(len(cnts_sort)):
            if(cnt_is_card[c]==1):

                corners = get_corners(cnts_sort[c])
                    
                team = game_logic.find_team(corners)
                if(team=="team1"):
                    cv2.drawContours(frame, cnts_sort[c], -1, (255, 0, 0), 3)  
                else:
                    cv2.drawContours(frame, cnts_sort[c], -1, (0, 255, 0), 2)

                # Compute similarity and return name of each card
                original = compare_contours(binary_frame, corners, cards_normal, cards_db_preprocessed)

                if original != None:
                    cv2.putText(frame,original[20:][:-4],corners[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                if num_cards_on_table==1:
                    assistir = original[20:][:-4]

                if num_cards_on_table!=4:
                    cv2.putText(frame,"Assistir: " + assistir,(5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                if num_cards_on_table==4:
                    #winning_team, winning_team_points = game_logic.load_game_logic(trunfo, assistir, card_team1_player1, card_team1_player2, card_team2_player1, card_team2_player2)
                    cv2.putText(frame,"Terminou a ronda",(5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#test_on_image()
main()