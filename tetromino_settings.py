TETROMINO_COLORS = {
    1: (15,155,215), #I
    2: (33,65,198), #J
    3: (227,91,2), #L
    4: (227,159,2), #O
    5: (89,177,1), #S
    6: (175,41,138), #T
    7: (215,15,55),  #Z
    8: (129, 133, 137)#Wall
}

GHOST_COLORS = {}
for i in range(1, len(TETROMINO_COLORS)):
    GHOST_COLORS[i+10] = tuple(map(lambda x : x // 2, TETROMINO_COLORS[i]))
    
for k,v in list(GHOST_COLORS.items()):
    TETROMINO_COLORS[k] = v
    