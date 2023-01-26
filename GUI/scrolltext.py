   
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets
class scrolltext(QtWidgets.QTextEdit):

    def __init__(self, parent=None):
        QtWidgets.QSlider.__init__(self, parent)
 

    def wheelEvent(self, event):
        self.emit(pyqtSignal("scrol(int)"), event.delta()/120)
        print("e")
