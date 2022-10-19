import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
import sys

from camera_calibration import calibrate_camera 
import game_logic
from projection import draw_trophy


NUM_CARDS = 4
CARD_MAX_AREA = 20000
CARD_MIN_AREA = 2000

####### Compare captured contours with card of the database

def compare_contours(captured_image, cards_corners, cards_db_names, cards_db):

    # Get corners and dimensions of the card on the database
    height, width = cards_db[0].shape[:2]

    # Initial original image and dst corners
    original = np.array(cards_corners, np.float32)
    dst = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], np.float32)

    less_difference = 150000
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
        if(len(original)==4):
            matrix = cv2.getPerspectiveTransform(original, dst)
        else: return

        # Generate frontal view from perspective
        card_frontal_view = cv2.warpPerspective(captured_image, matrix, (width, height))

        # Compare images with cards on the database
        for c in range(len(cards_db_names)):
            similarity = compute_similarity(card_frontal_view, cards_db[c])
            if similarity < less_difference:
                less_difference = similarity
                best_match = cards_db_names[c]
                diff_img = cv2.absdiff(card_frontal_view, cards_db[c])
                #cv2.imshow("diff",diff_img)
        
    return best_match



####### Compare captured contours of superpositioned cards with cards of the database

def compare_contours_superpositions(captured_image, cards_db_names, cards_db):
    sift = cv2.xfeatures2d.SIFT_create() 
    index_params = dict(algorithm = 0, trees = 5) 
    search_params = dict() 
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    _, desc_grayframe = sift.detectAndCompute(captured_image, None) 

    best_match = 0
    best_card = ""
    for c in range(len(cards_db_names)):
        kp_image, desc_image =sift.detectAndCompute(cards_db[c], None)  
        matches= flann.knnMatch(desc_image, desc_grayframe, k=2) 

        good_points=[] 
        for m, n in matches: 
            if(m.distance < 0.6*n.distance): 
                good_points.append(m) 

        if(len(good_points)>best_match): 
            best_match=len(good_points)
            best_card = cards_db_names[c]
    print(best_card)
    return best_card



####### Get the corners of a card through its countor 

def get_corners(contour):
    return cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)



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



####### Binarize an image

def binarize_image(original_image):
    gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img_w, img_h = np.shape(original_image)[:2]

    bkg_level = gray_scale_image[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + 100

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
    cnt_is_card = np.zeros(len(cnts),dtype=int) # [0,1,0,1,1,...] where 1 if is a card and 2 if there are two cards superposition

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        #if (size > CARD_MIN_AREA) and (len(approx) == 8 or len(approx) == 6): # Two cards overlapping
            #cnt_is_card[i] = 2

        if ( (size < CARD_MAX_AREA) and ( size > CARD_MIN_AREA) and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1
        
    return cnts_sort, cnt_is_card


#### Find the marker

def find_marker(thresh_image, template):
    height, width = template.shape[:2]
    cnts,_ = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        peri = cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
        if(len(approx)) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if  (0.8<=(float(w)/h)<=1.2):
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

    # Get camara image
    cap = cv2.VideoCapture(0)

    # Calibrate camera:
    ret, cameraMatrix, dist, rvecs, tvecs = calibrate_camera()

    # Pre-process database cards:
    cards_normal = glob.glob('images/cards_normal/*.png')
    cards_db_preprocessed = []

    for card_name in cards_normal:
        image = cv2.imread(card_name)
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_scale_image, 127, 255, cv2.THRESH_BINARY)
        cards_db_preprocessed.append(binary_image)

    # Pre-process marker image:
    marker = cv2.imread('images/marker.png')
    gray_scale_marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
    _, marker = cv2.threshold(gray_scale_marker, 100, 255, cv2.THRESH_BINARY)

    # Game Logic:
    assistir = ""
    card_team1_player1 = ""
    card_team1_player2 = ""
    card_team2_player1 = ""
    card_team2_player2 = ""
    points_team_1 = 0
    points_team_2 = 0
    new_round = True

    while True:
        _, frame = cap.read()
        h, w = np.shape(frame)[:2]

        cv2.rectangle(frame, (10,h-8), (w-15, h-85), (255,255,255), -1)
        cv2.rectangle(frame, (10,h-8), (w-15, h-85), (0, 0, 0), 2)

        # Binarize frame
        binary_frame = binarize_image(frame)
        
        # Find and recognize marker
        marker_corners = find_marker(binary_frame,marker)
        draw_trophy(frame, marker_corners, cameraMatrix, dist)

        # Find cards
        cnts_sort, cnt_is_card = find_cards(binary_frame)
        num_cards_on_table = np.count_nonzero(cnt_is_card==1) 

        cv2.putText(frame,"TRUNFO: " + trunfo,(20,h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX                 )
        cv2.putText(frame,"TEAM 1" ,(300,h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 153), 2, cv2.FONT_HERSHEY_SIMPLEX                 )
        cv2.putText(frame,str(points_team_1) + " POINTS",(300,h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 153), 2, cv2.FONT_HERSHEY_SIMPLEX                 )
        cv2.putText(frame,"TEAM 2" ,(500,h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX                 )
        cv2.putText(frame,str(points_team_2) + " POINTS",(500,h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX                 )
        cv2.putText(frame,"Press Q to QUIT the game",(20,h-13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 153), 1, cv2.FONT_HERSHEY_SIMPLEX                 )

        for c in range(len(cnts_sort)):
            original = ""
            corners=[[]]

            if(cnt_is_card[c]==1): # is just a card
                corners = get_corners(cnts_sort[c])

                # Find card team:
                team = game_logic.find_team(corners)
                
                if team=="team1":
                    cv2.drawContours(frame, cnts_sort, c, (0, 0, 255), 2)  
                elif team=="team2":
                    cv2.drawContours(frame, cnts_sort, c, (255, 0, 0), 2)
                    
                # Compute similarity and return name of each card
                original = compare_contours(binary_frame, corners, cards_normal, cards_db_preprocessed)

            # Two cards overlapping
            #if(cnt_is_card[c]==2):
                #cv2.drawContours(frame, cnts_sort[c], -1, (0, 255, 255), 2)  
                #original = compare_contours_superpositions(binary_frame, cards_normal, cards_db_preprocessed)

            if c==0: card_team1_player1 = original
            elif c==1: card_team2_player1 = original
            elif c==2: card_team1_player2 = original
            elif c==3: card_team2_player2 = original

            if original is not None and original != "" and len(corners[0])>0:
                cv2.putText(frame,original[20:][:-4],corners[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.FONT_HERSHEY_SIMPLEX)
                
        
        if num_cards_on_table==0:
            cv2.putText(frame,"You can start to play" + assistir,(20,h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)

        if num_cards_on_table==1 and original != None:
            new_round = True
            assistir = original[20:][:-4]
            cv2.putText(frame,"Card to assist: " + assistir,(20,h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)

        if num_cards_on_table==4:
            if card_team1_player1 is not None and card_team1_player2 is not None and card_team2_player1 is not None and card_team2_player2 is not None:
                winning_team, points = game_logic.load_game_logic(trunfo, assistir, card_team1_player1, card_team1_player2, card_team2_player1, card_team2_player2)
                if winning_team=="team1" and new_round==True:
                    points_team_1 += points
                    new_round = False
                elif winning_team=="team2" and new_round==True: 
                    points_team_2 += points
                    new_round = False
            cv2.putText(frame,"Finished round",(20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        

        cv2.imshow('Sueca Game Assistant', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()