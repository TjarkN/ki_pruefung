#! /usr/bin/python
#-*- coding: utf-8 -*-


from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets



class radarField(QtWidgets.QGraphicsItemGroup):
    
    def __init__(self, qPointList, bot, rType):
        QtWidgets.QGraphicsItemGroup.__init__(self)
        self.rType = rType
        if rType == "poly":
            self.item = QtWidgets.QGraphicsPolygonItem()
            self.robot = bot
            self.polygon = QtGui.QPolygonF(qPointList)
            self.item.setPolygon(self.polygon)

        elif rType == "round":
            self.item = QtWidgets.QGraphicsEllipseItem()
            self.robot = bot
            self.item.setRect(qPointList[0], qPointList[1],qPointList[2],qPointList[3])

        color = QtGui.QColor(255, 100, 6, 10)
        brush = QtGui.QBrush(color)
        pen = QtGui.QPen(color)
        self.item.setBrush(brush)
        self.item.setPen(pen)
        self.addToGroup(self.item)
            
            
    def setVisible(self, bol):
        if bol:
            color = QtGui.QColor(255, 100, 6, 15)
        else:
            color = QtGui.QColor(255, 100, 6, 0)
        brush = QtGui.QBrush(color)
        pen = QtGui.QPen(color)
        self.item.setBrush(brush)
        self.item.setPen(pen)
    
        
        
