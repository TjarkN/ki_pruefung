#! /usr/bin/python
#-*- coding: utf-8 -*-

from PyQt5.QtGui import QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QMessageBox, QGraphicsRectItem
from PyQt5.QtCore import pyqtSignal,  QPointF
from PyQt5 import QtCore
from Objects.robot import Robot
import time,  os,  random
from GUI.outPrint import outPrint

class Graph(QGraphicsScene):
    
    def __init__(self,  parent, width,  height):
        QGraphicsScene.__init__(self,  parent)
        self.setSceneRect(0, 0, width, height)
        self.Parent = parent
        
        #self.Parent.graphicsView.centerOn(250, 250)
        self.width = width
        self.height = height
        self.grid = self.getGrid()
        self.setTiles()
        self.action_counter = 0
        self.max_action_per_battle = None #100

    def add_action_counter(self):
        self.action_counter += 1
        if self.max_action_per_battle is not None and self.action_counter > self.max_action_per_battle:
            self.battleFinished(True)

    def AddRobots(self, botList, models = None, trainings = None):
        
        """
        """
        self.aliveBots = []
        self.deadBots = []
        try:
            posList = random.sample(self.grid, len(botList))
            identity = 0
            for bot in botList:
                try:
                    model_bot = None
                    training_bot = None
                    if models is not None:
                        model_bot = models[identity]
                        training_bot = trainings[identity]
                    robot = bot(self.sceneRect().size(), self, str(bot), identity, model_bot, training_bot)
                    #if training_bot is not None:
                    #    training_bot.rl_bot = robot
                    self.aliveBots.append(robot)
                    self.addItem(robot)
                    robot.setPos(posList.pop())
                    """
                    if identity == 0:
x                        robot.setPos(50, 50)
                    else:
                        robot.setPos(50, 200)
                    """
                    self.Parent.addRobotInfo(robot)
                    identity += 1
                except Exception as  e:
                    print("Problem with bot file '%s': %s" % (bot, str(e)))
            self.Parent.battleMenu.close()
        except ValueError:
            QMessageBox.about(self.Parent, "Alert", "Too many Bots for the map's size!")
        except AttributeError:
            pass

    def battleFinished(self, timeout = False):
        print("battle terminated")
        for bot in self.aliveBots:
            if bot.training is not None:
                bot.training.train(game_over=True)
                return
        try:
            self.deadBots.append(self.aliveBots[0])
            self.removeItem(self.aliveBots[0])
            self.action_counter=0
        except IndexError:
            pass
        j = len(self.deadBots)
        
        if not timeout:
            for i in range(j):
                print("N°",  j - i , ":", (self.deadBots[i]))
                if j-i == 1: #first place
                    self.Parent.statisticDico[repr(self.deadBots[i])].first += 1
                if j-i == 2: #2nd place
                    self.Parent.statisticDico[repr(self.deadBots[i])].second += 1
                if j-i ==3:#3rd place
                    self.Parent.statisticDico[repr(self.deadBots[i])].third += 1

                self.Parent.statisticDico[repr(self.deadBots[i])].points += i
                
        self.Parent.chooseAction()       

                    
    def setTiles(self):
        #background
        brush = QBrush()
        pix = QPixmap(os.getcwd() + "/robotImages/tile.png")
        brush.setTexture(pix)
        brush.setStyle(24)
        self.setBackgroundBrush(brush)
        
        #wall
        #left
        left = QGraphicsRectItem()
        pix = QPixmap(os.getcwd() + "/robotImages/tileVert.png")
        left.setRect(QtCore.QRectF(0, 0, pix.width(), self.height))
        brush.setTexture(pix)
        brush.setStyle(24)
        left.setBrush(brush)
        left.name = 'left'
        self.addItem(left)
        #right
        right = QGraphicsRectItem()
        right.setRect(self.width - pix.width(), 0, pix.width(), self.height)
        right.setBrush(brush)
        right.name = 'right'
        self.addItem(right)
        #top
        top = QGraphicsRectItem()
        pix = QPixmap(os.getcwd() + "/robotImages/tileHori.png")
        top.setRect(QtCore.QRectF(0, 0, self.width, pix.height()))
        brush.setTexture(pix)
        brush.setStyle(24)
        top.setBrush(brush)
        top.name = 'top'
        self.addItem(top)
        #bottom
        bottom = QGraphicsRectItem()
        bottom.setRect(0 ,self.height - pix.height() , self.width, pix.height())
        bottom.setBrush(brush)
        bottom.name = 'bottom'
        self.addItem(bottom)
        
    def getGrid(self):
        w = int(self.width/80)
        h = int(self.height/80)
        l = []
        for i in range(w):
            for j in range(h):
                l.append(QtCore.QPointF((i+0.5)*80, (j+0.5)*80))
        return l
