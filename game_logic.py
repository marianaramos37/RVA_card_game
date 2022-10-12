from tkinter.tix import TList
import numpy as np


def load_game_logic(trunfo,assistir,card_t1_p1,card_t1_p2,card_t2_p1,card_t2_p2):
    winning_team = "team1"
    
    points_team1 = get_points(card_t1_p1)+get_points(card_t1_p2)
    points_team2 = get_points(card_t2_p1)+get_points(card_t2_p2)

    naipe = assistir[-4:] 

    if(points_team1 > points_team2):
        winning_team = "team1"
    else:
        winning_team = "team2"

    # Todas as equipas assistirem
    #if card_t1_p1[:4]==naipe and card_t1_p2[:4]==naipe and card_t2_p1[:4]==naipe and card_t2_p2[:4]==naipe:
    #    poits_team1 = get_points(card_t1_p1)+get_points(card_t1_p2)
    #    poits_team2 = get_points(card_t2_p1)+get_points(card_t2_p1)
    #    if(poits_team1 > poits_team2):
    #        winning_team = "team1"
    #    else:
    #        winning_team = "team2"
    #if apenas uma equipa jogar trunfo
        # pontos vao todos para essa equipa
    #if ambas as equipas jogarem trunfos   
        # pontos vao para a equipa com trunfo maior

    return winning_team, points_team1, points_team2

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
