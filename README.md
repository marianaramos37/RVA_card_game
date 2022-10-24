# RVA_card_game

This application has the aim of assisting the playing of a simple card game with augmented reality techniques. In our case, the game being assisted is Sueca. Sueca is a very
popular card game in Portugal played by four people (two teams of two
players) where there are 120 points up for grabs and the pair that scores
more than 60 points wins.

The program developed will assist the
game by showing the number of points of each team in each round. By
the end of the match, a trophy with the faces of the winning team will
show up at the center of the table indicating which team won.

## Run the project

To run the project simply use the command:
```
python src/card_detector.py [TRUNFO]
```
Where [TRUNFO] can be one of the following: 'copas', 'paus', 'ouros', 'espadas'

## Implementation details

The implementation details can be found in the report, inside the folder docs, as well as a demonstration video.

### File structure

- *src Folder*:
    - **camera_calibration.py**: Function runed in the begining of each execution to get the intrinsic parameters of the camera;
    - **card_detector.py**: Main game loop with functions regarding the card positioning and recognition;
    - **augmented_test.py**: Drawing of the trophy in the mark.
- *docs Folder*:
    - **demo_video**: demostration video of the project running;
    - **Report_Group4**: report with implementation details of the project.
- *images Folder*:
    - **cards_normal**: card deck used in the recognition process;
    - **auxiliar_images**: auxiliar images to keep track of the devlelopment of the project;
    - **chessboard_calibration**: images used in the camera calibration phase.

