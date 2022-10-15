from tkinter.tix import TList
import numpy as np


def load_game_logic(trunfo,assistir,card_t1_p1,card_t1_p2,card_t2_p1,card_t2_p2):
    winning_team = ""
    usedTrunfo = False
    cards_played = [card_t1_p1, card_t1_p2, card_t2_p1, card_t2_p2]
    cards_to_consider = np.zeros(4, dtype=int)
    points = get_points(card_t1_p1)+get_points(card_t1_p2)+get_points(card_t2_p1)+get_points(card_t2_p2)
    suit = getSuit(assistir)
    
    if (getSuit(card_t1_p1) == trunfo):
        cards_to_consider[0] = 1
        usedTrunfo = True
    if (getSuit(card_t1_p2) == trunfo):
        cards_to_consider[1] = 1
        usedTrunfo = True
    if (getSuit(card_t2_p1) == trunfo):
        cards_to_consider[2] = 1
        usedTrunfo = True
    if (getSuit(card_t2_p2) == trunfo):
        cards_to_consider[3] = 1
        usedTrunfo = True

    if (usedTrunfo == False):
        if (getSuit(card_t1_p1) == suit):
            cards_to_consider[0] = 1
        if (getSuit(card_t1_p2) == suit):
            cards_to_consider[1] = 1
        if (getSuit(card_t2_p1) == suit):
            cards_to_consider[2] = 1
        if (getSuit(card_t2_p2) == suit):
            cards_to_consider[3] = 1
    
    bestCard = ""
    for i in range(4):
        if (cards_to_consider[i] == 1 and (bestCard == "" or isBetter(cards_played[i], bestCard))):
            bestCard = cards_played[i]
            if (i < 2): winning_team == "team1"
            else: winning_team == "team2"

    return winning_team, points

# Find team: Horizontal cards are team 2 vertical are team 1
def find_team(card_corners): 
        # top left point: min(x+y)
        tl = sorted(card_corners, key=lambda p: (p[0][0]) + (p[0][1]))[0] 
        # top right point: max(x-y)
        tr = sorted(card_corners, key=lambda p: (p[0][0]) - (p[0][1]))[-1]

        corners_rest = np.delete(card_corners, np.where(card_corners == tr), axis=0)

        bl = tl
        if len(corners_rest)>0:
            bl = sorted(corners_rest, key=lambda p: (p[0][0]))[0] 
        
        # Card is vertical
        if (bl[0][1] - tl[0][1]) > (tr[0][0] - tl[0][0]):
            return "team2"
        else:  # Card is horizontal
            return "team1"
            
def get_points(card):
    points = 0
    card=card[20:]
    if card!="":
        if card[0]=="a":
            points = 11
        elif card[0]=="7":
            points = 10
        elif card[0]=="r":
            points = 4
        elif card[0]=="v":
            points = 3
        elif card[0]=="d":
            points = 2
        else: points = 0
    return points

def getSuit(firstCard):
    termination = firstCard[-7:-4]
    suit = ""
    if (termination == "das"):
        suit = "espadas"
    elif (termination == "aus"):
        suit = "paus"
    elif (termination == "pas"):
        suit = "copas"
    elif (termination == "ros"):
        suit = "ouros"
    return suit
    
def isBetter(cardA, cardB):
    if (getValue(cardA) >= getValue(cardB)): return True
    return False
    
def getValue(card):
    if (card[0] == "a"): return 15
    if (card[0] == "7"): return 14
    if (card[0] == "r"): return 13
    if (card[0] == "v"): return 12
    if (card[0] == "d"): return 11
    return int(card[0])
    