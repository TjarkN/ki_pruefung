#! /usr/bin/python
# -*- coding: utf-8 -*-
from AI.actions import Action
from Objects.robot import Robot  # Import a base Robot
from keras.models import model_from_json
import numpy as np


class ReinforcementLearningBotMinimal(Robot):
    # Create a Robot

    def init(self):
        # To initialise your robot
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
        self.last_action = None
        self.last_input = None

        if self.model is None:
            json_file = open('model_YourBotName.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("model_YourBotName.h5")

    def randmax(self, values):
        max_values = []
        current_max = values[0]
        index = 0
        for v in values:
            if v > current_max:
                max_values = [index]
                current_max = v
            elif v == current_max:
                max_values.append(index)
            index += 1
        if len(max_values) == 0:
            return np.random.randint(0,len(values)-1)
        else:
            return np.random.choice(max_values)

    def run(self):
        # main loop to command the bot
        input_t = self.observe()
        if self.training and self.last_action is not None:
            game_over = False
            self.training.train(game_over)

        q = self.model.predict(input_t)
        # Select the action with the highest expected reward
        action = self.randmax(q[0])

        self.last_action = action
        self.last_input = input_t

        if action == 0:
            self.turn_right()
        elif action == 1:
            self.turn_left()
        elif action == 2:
            self.forward()
        elif action == 3:
            self.backwards()
        elif action == 2:
            self.shoot()

    def calcAngleTo_singed(self, pos_self, pos_other):
        test_step_to_left = 10
        gun_heading_to_left = self.getGunHeading() - test_step_to_left
        angle_to_enemy = np.round(Action.angleTo(pos_self, pos_other, self.getGunHeading()), 2)
        angle_to_enemy_to_left = np.round(Action.angleTo(pos_self, pos_other, gun_heading_to_left), 2)

        if (abs(angle_to_enemy - angle_to_enemy_to_left) >= test_step_to_left \
            and (angle_to_enemy_to_left > angle_to_enemy)) \
                or (abs(angle_to_enemy - angle_to_enemy_to_left) < test_step_to_left \
                    and angle_to_enemy >= 90):
            angle_to_enemy *= -1
        return angle_to_enemy

    def observe(self):
        pos_self = (self.getPosition().x(), self.getPosition().y())
        if self.getPosition_enemy() is not None:
            pos_enemy = (self.getPosition_enemy().x(), self.getPosition_enemy().y())
            angle_to_enemy = self.calcAngleTo_singed(pos_self, pos_enemy)
            return np.array([angle_to_enemy/ 360]).reshape((1, -1))
        else:
            return np.array([self.last_input[0, 0]]).reshape((1, -1))

    def onHitWall(self):
        self.reset()  # To reset the run function to the begining (automatically called on hitWall, and robotHit event)
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
        if self.training and self.last_action is not None:
            # here you might want to add things
            game_over = True
            self.training.train(game_over)

    def onTargetSpotted(self, botId, botName, botPos):  # NECESARY FOR THE GAME
        "when the bot see another one"
        self.rPrint("I see the bot:" + str(botId) + "on position: x:" + str(botPos.x()) + " , y:" + str(botPos.y()))

    def onEnemyDeath(self):
        """When an enemy dies"""
        if self.training and self.last_action is not None:
            #here you might want to add things
            game_over = True
            self.training.train(game_over)