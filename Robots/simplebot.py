#! /usr/bin/python
# -*- coding: utf-8 -*-
from AI.actions import Action
from AI.state import State

from Objects.robot import Robot  # Import a base Robot


class Simplebot(Robot):  # Create a Robot


    def init(self):  # To initialyse your robot

        # Feel free to customize: Set the bot color in RGB
        self.setColor(0, 0, 100)
        self.setGunColor(0, 0, 100)
        self.setRadarColor(0, 60, 0)
        self.setBulletsColor(255, 150, 150)
        self.maxDepth = 5

        #Don't Change
        self.setRadarField("thin")
        self.radarVisible(True)  # if True the radar field is visible
        self.gun_to_side()
        self.lockRadar("gun")
        size = self.getMapSize()

    def run(self):  # main loop to command the bot
        """
        to create your own bot, create an new python-file in python package "Robots" and giv it a name
        copy the code of simplebot and rename the class
        starting the main.py starts the game

        use this method to control the bot
        1. information from your sensors:
            - get your position: self.getPosition()
            - get enemies position: self.getPosition_enemy()
            - get left energy (max 100): self.energy_left_self()
            - get left energy of enemy: self.energy_left_enemy()
            - return true if a shot would hit the enemy: self.shot_possible_by_enemy()
            - return true if a shot by the enemy would hit you: self.shot_possible_at_enemy()
            - return the angle of your gun (looking down (0,1) equals an angle of 0: self.getGunHeading()
            - return the angle of your enemies gun (looking down (0,1) equals an angle of 0: self.getGunHeading_enemy()+
            - returns the map size (left upper corner is (0,0):  self.getMapSize
        2. possible actions
            - turns 45° to the right: self.turn_right()
            - turns 45° to the left: self.turn_left()
            - move 50 points forward: self.forward()
            - move 50 points backward: self.backwards()
            - shoots cost 3 points of energy (when the enemy is hit he gets 9 points damage and you receive 6 points cure): self.shoot()
        """

    def eval(self, state):
        """implement your evaluation function here"""
        utility = 0
        return self, utility


    def onHitWall(self):
        self.reset()  # To reset the run fonction to the begining (auomatically called on hitWall, and robotHit event)
        self.rPrint('ouch! a wall !')

    def sensors(self):  # NECESARY FOR THE GAME
        pass

    def onRobotHit(self, robotId, robotName):  # when My bot hit another
        self.rPrint('collision with:' + str(robotId))

    def onHitByRobot(self, robotId, robotName):
        self.rPrint("damn a bot collided me!")

    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):  # NECESARY FOR THE GAME
        """ When i'm hit by a bullet"""
        self.rPrint("hit by " + str(bulletBotId) + "with power:" + str(bulletPower))

    def onBulletHit(self, botId, bulletId):  # NECESARY FOR THE GAME
        """when my bullet hit a bot"""
        self.rPrint("fire done on " + str(botId))

    def onBulletMiss(self, bulletId):  # NECESARY FOR THE GAME
        """when my bullet hit a wall"""
        self.rPrint("the bullet " + str(bulletId) + " fail")

    def onRobotDeath(self):  # NECESARY FOR THE GAME
        """When my bot die"""
        self.rPrint("damn I'm Dead")

    def onTargetSpotted(self, botId, botName, botPos):  # NECESARY FOR THE GAME
        "when the bot see another one"
        self.rPrint("I see the bot:" + str(botId) + "on position: x:" + str(botPos.x()) + " , y:" + str(botPos.y()))

    def onEnemyDeath(self):
        pass