# -*- coding: utf-8 -*-

"""
Module implementing outPrint.
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot,  pyqtSignal

from GUI.Ui_outPrint import Ui_Form

class outPrint(QWidget, Ui_Form):
    """
    Class documentation goes here.
    """
    def __init__(self, parent = None):
        """
        Constructor
        """
        QWidget.__init__(self, parent)
        self.setupUi(self)

        
    def add(self, msg):
        self.textEdit.append(msg)
