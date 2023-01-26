from GUI.window import MainWindow
from PyQt5 import QtWidgets
import sys
import os

sys.path.append(os.getcwd() + "/GUI")
sys.path.append(os.getcwd() + "/Objects")
sys.path.append(os.getcwd() + "/robotImages")
sys.path.append(os.getcwd() + "/Robots")

class GameEnvironment():

    def __init__(self, width, height, initial_speed = 0, no_grafics = True):
        self.app = QtWidgets.QApplication(sys.argv)
        self.app.setApplicationName("Python-Robocode")
        self.window = MainWindow(initial_speed = initial_speed)
        if not no_grafics:
            self.window.show()
        self.width = width
        self.height = height

    def start(self, botList, models = None, trainings = None):
        self.window.setUpBattle(self.width, self.height, botList, models, trainings, True)
        sys.exit(self.app.exec_())


    def restart(self, botList, models = None, trainings = None):
        self.window.startBattle(botList, models, trainings)
