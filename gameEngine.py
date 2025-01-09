from collections import deque
import random
from gameComponents.field import Field
from gameComponents.render import Renderer
from gameComponents.controller import Controller
from gameComponents.piece import Piece, SHAPE_ID
from pygame import KEYDOWN, KEYUP, K_LEFT, K_RIGHT, K_z, K_x, K_c, K_SPACE, K_UP, K_DOWN
from tetromino_settings import *
from window_settings import *
import numpy as np


debug = False

"""
Spawns and moves the falling tetromino
"""
class GameEngine:
    def __init__(self):
        self.field = None
        self.stats = None
        self.renderer = None
        self.ctrl = None
        
    
    def reset(self):
        self.renderer = Renderer()
        self.field = Field()
        self.ctrl = Controller()
        self.field.reset()
        self.ctrl.reset()
        self.renderer.reset()
        
        
        self.gen = self.pieceGen()
        self.pieceQueue = deque()
        for i in range(NUM_PREVIEW):
            self.pieceQueue.append(next(self.gen))
            
            
        self.spawnNewPiece()
        self.dasDirection = 0
        self.holdPiece = None
        self.holdAllowed = True
        self.dasCount = 0        
        self.ghostEnabled = True

        
        if debug:
            self.currentPiece = Piece(SHAPE_ID.S)
            self.pieceQueue = deque()
            self.pieceQueue.append(SHAPE_ID.S)
            self.pieceQueue.append(SHAPE_ID.T)
            self.pieceQueue.append(SHAPE_ID.T)
            self.pieceQueue.append(SHAPE_ID.T)
            self.pieceQueue.append(SHAPE_ID.T)
            self.pieceQueue.append(SHAPE_ID.T)
            self.pieceQueue.append(SHAPE_ID.S)
            self.pieceQueue.append(SHAPE_ID.T)

            self.field.setupPredefinedBoard()
        
    
    def pieceGen(self):
        while True:
            bag = list(SHAPE_ID)
            random.shuffle(bag)
            for piece in bag:
                yield piece


    def update(self):
        self.ctrl.updateTime()
        
        ## Each logic sets these boolean variables, which are evaluated later using priorities
        move = False
        rotate = False
        hardDrop = False
        softDrop = False
        naturalDrop = False
        hold = False
        lock = False
        gameOver = False
    
        ## Movement logic -- works good 2025 01 07
        moveDirection = self.getMoveDirection()
        if moveDirection == self.dasDirection:
            if self.dasCharged():
                move = True
                self.dasCount -= ARR
            else:
                self.dasCount += 1
        else:
            move = True
            self.dasCount = 0
            self.dasDirection = moveDirection
            
            
        if self.ctrl.isPush(K_DOWN):
            softDrop = True
            self.softDropTimer = 0
        elif self.ctrl.isPress(K_DOWN):
            if self.softDropTimer == SOFTDROP_FRAMES:
                softDrop = True
                self.softDropTimer = 0
            else:
                self.softDropTimer += 1
        else:
            self.softDropTimer = 0
            
            
        rotateDirection = 0    
        if self.ctrl.isPush(K_z):
            rotate = True
            rotateDirection = -1
        if self.ctrl.isPush(K_x):
            rotate = True
            rotateDirection = 1
            
            
        if self.ctrl.isPush(K_SPACE):
            hardDrop = True
            
        
        if self.holdAllowed and self.ctrl.isPush(K_c):
            hold = True
                        
        
        if self.gravityCharged():
            naturalDrop = True
            self.naturalDropTimer = 0
        else:
            self.naturalDropTimer += 1

        
        if self.currentPiece.checkCollision(self.pieceX, self.pieceY - 1, self.field):
            self.lockTimer += 1
            if self.lockTimer == FRAMES_TIL_LOCK:
                lock = True
        else:
            self.lockTimer = 0
        
        
        if hold:
            self.holdAllowed = False
            self.swapHoldPiece()
            move = False
            hardDrop = False
            softDrop = False
            naturalDrop = False
            lock = False
            rotate=False
        
        
        if hardDrop:
            while(not self.currentPiece.checkCollision(self.pieceX, self.pieceY - 1, self.field)):
                self.pieceY -= 1
            lock = True
            move = False
            naturalDrop = False
            softDrop = False
            hold = False
            rotate = False
            
        
        if lock:
            if self.currentPiece.checkCollision(self.pieceX, self.pieceY, self.field):
                print("game over")
                gameOver = True
                move = False
                hardDrop = False
                softDrop = False
                naturalDrop = False
                hold = False
                lock = False
                rotate = False
            else:
                self.applyPiece(self.currentPiece, self.pieceX, self.pieceY, self.field)
                self.field.clearLine()
                self.spawnNewPiece()
                move = False
                rotate = False
                naturalDrop = False
                softDrop = False
                hold = False
                
        
        if gameOver:
            self.finishGame()
            return gameOver
                
                
        if rotate:
            success, offset = self.currentPiece.rotate(rotateDirection, self.pieceX, self.pieceY, self.field)
            if success:
                self.pieceX += offset[0]
                self.pieceY += offset[1]
                
        if move:
            self.move(moveDirection)
        
        if naturalDrop:
            self.drop()
            
        if softDrop:
            self.drop()

        if self.ghostEnabled:
            self.renderer.render_frame(self.field, 
                                    self.pieceQueue, 
                                    self.holdPiece, 
                                    self.currentPiece, 
                                    self.pieceX, 
                                    self.pieceY,
                                    self.currentPiece.ghostCoordinates(self.pieceX, self.pieceY, self.field))
        else:
            self.renderer.render_frame(self.field, 
                        self.pieceQueue, 
                        self.holdPiece, 
                        self.currentPiece, 
                        self.pieceX, 
                        self.pieceY,
                        None)

        return gameOver
    
    
    def swapHoldPiece(self):
        if self.holdPiece is None:
            self.holdPiece = self.currentPiece.getType()
            self.spawnNewPiece()
            self.holdAllowed = False
        else:
            temp = self.holdPiece
            self.holdPiece = self.currentPiece.getType()
            self.currentPiece = Piece(temp)
            if temp is SHAPE_ID.I:
                self.pieceX = SPAWN_COORDINATE_X - 1
                self.pieceY = SPAWN_COORDINATE_Y - 1
            else: 
                self.pieceX = SPAWN_COORDINATE_X
                self.pieceY = SPAWN_COORDINATE_Y
            self.currentPieceRotationCount = 0
            self.naturalDropTimer = FRAMES_TIL_FALL
            self.softDropTimer = 0
            self.hardDropTimer = 0
            self.lockTimer = 0
            self.holdAllowed = False
            
    
    def spawnNewPiece(self):
        self.currentPiece = Piece(self.pieceQueue.popleft())
        self.pieceQueue.append(next(self.gen))
        if self.currentPiece.getType() is SHAPE_ID.I:
            self.pieceX = SPAWN_COORDINATE_X - 1
            self.pieceY = SPAWN_COORDINATE_Y - 1
        else: 
            self.pieceX = SPAWN_COORDINATE_X
            self.pieceY = SPAWN_COORDINATE_Y
        self.currentPieceRotationCount = 0
        self.naturalDropTimer = FRAMES_TIL_FALL
        self.softDropTimer = 0
        self.hardDropTimer = 0
        self.lockTimer = 0
        self.holdAllowed = True
    
    
    def applyPiece(self, piece : Piece, x, y, field : Field):
        for i in range(4):
            tileX = x + piece.getTileX()[i]
            tileY = y + piece.getTileY()[i]
            field.setValue(tileX, tileY, piece.type.value)
    
    
    def getMoveDirection(self):
        if self.ctrl.isPress(K_LEFT) and self.ctrl.isPress(K_RIGHT):
            if self.ctrl.getButtonTime(K_LEFT) < self.ctrl.getButtonTime(K_RIGHT):
                return -1
            else: 
                return 1
        elif self.ctrl.isPress(K_LEFT):
            return -1
        elif self.ctrl.isPress(K_RIGHT):
            return 1
        else:
            return 0
    
    
    def gravityCharged(self):
        return self.naturalDropTimer == FRAMES_TIL_FALL
    
    
    def dasCharged(self):
        return self.dasCount == DAS
    
    
    def resetDAS(self):
        self.dasCount = 0
    
    
    def move(self, dir):
        if not self.currentPiece.checkCollision(self.pieceX + dir, self.pieceY, self.field):
            self.pieceX += dir     
            
    
    def drop(self):
        if not self.currentPiece.checkCollision(self.pieceX, self.pieceY - 1, self.field):
            self.pieceY -= 1
            
            
    def finishGame(self):
        self.field.greyfyBoard()
        self.renderer.render_frame(self.field, self.pieceQueue, self.holdPiece, None, None, None, None)

    
    def close(self):
        self.renderer.close()
        
        self.field = None
        self.ctrl = None
        self.stats = None
        self.currentPiece = None
        self.pieceX = 0
        self.pieceY = 0
        self.currentPieceRotationCount = 0
        
        self.holdPiece = None
        
        self.dasCount = 0
        self.dasDirection = 0
        self.dasRepeat = False
        self.dasInstant = False
        
        self.softdropTimer = 0
        self.hardDropTimer = 0
        
        self.ghost = False
        
        