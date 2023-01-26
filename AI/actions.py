import numpy as np
import math

class Action:

    #__actions = {"turn_right": 5, "turn_left": -5, "shoot": 3}
    __actions = {"turn_right": 45, "turn_left": -45, "forward": 50, "backward": -50, "shoot": 3}
    #for Minimax __actions = [["turn_right", 45], ["turn_left", -45], ["forward", 50], ["backward", -50], ["shoot", 3]]
    __pause = 10

    @classmethod
    def get_actions(self):
        return self.__actions

    @classmethod
    def get_turn_right(self):
        return self.__actions.get("turn_right")

    @classmethod
    def get_turn_left(self):
        return self.__actions.get("turn_left")

    @classmethod
    def get_forward(self):
        return self.__actions.get("forward")

    @classmethod
    def get_backward(self):
        return self.__actions.get("backward")

    @classmethod
    def get_shoot(self):
        return self.__actions.get("shoot")

    @classmethod
    def get_pause(self):
        return self.__pause

    @staticmethod
    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        l_v1 = np.linalg.norm(v1)
        l_v2 = np.linalg.norm(v2)
        if l_v1 == 0 or l_v2 == 0:
            print("error: v1 or v2 has length zero")
            return 0
        v1_u = v1 / l_v1
        v2_u = v2 / l_v2
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @classmethod
    def faces(self, pos_1, pos_2, angle_1):
        dir_enemy = (pos_2[0] - pos_1[0], pos_2[1] - pos_1[1])
        if dir_enemy == (0,0):
            return False
        heading_self = (angle_1 / 360 - angle_1 // 360) * 360
        heading_self_r = heading_self * math.pi / 180
        dir_self = (-math.sin(heading_self_r), math.cos(heading_self_r))
        angle_enemy = self.angle_between(dir_self, dir_enemy)
        angle_enemy = 180 / math.pi * angle_enemy

        alpha = math.atan(25/math.sqrt(dir_enemy[0]*dir_enemy[0]+dir_enemy[1]*dir_enemy[1]))
        alpha = math.degrees(alpha)
        if -alpha <= angle_enemy <= alpha:
            return True
        else:
            return False

    @classmethod
    def angleTo(self, pos_1, pos_2, angle_1):
        dir_enemy = ( pos_2[0] - pos_1[0], pos_2[1] - pos_1[1])
        heading_self = (angle_1 / 360 - angle_1 // 360) * 360
        heading_self_r = heading_self * math.pi / 180
        dir_self = (-math.sin(heading_self_r), math.cos(heading_self_r))
        angle_enemy = self.angle_between(dir_self, dir_enemy)
        angle_enemy = 180 / math.pi * angle_enemy
        return angle_enemy

    @classmethod
    def distance_between(self, pos_1, pos_2):
        dir_enemy = (pos_2[0] - pos_1[0], pos_2[1] - pos_1[1])
        return math.sqrt(dir_enemy[0]*dir_enemy[0]+dir_enemy[1]*dir_enemy[1])