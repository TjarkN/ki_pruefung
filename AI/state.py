from AI.actions import Action
import math


class State:

    def __init__(self, energy_self, energy_enemy, shot_possibly_by_enemy, shot_possibly_at_enemy, pos_self, pos_enemy,
                 angle_self, angle_enemy, map_size):
        self.energy_self = energy_self
        self.energy_enemy = energy_enemy
        self.shot_possibly_by_enemy = shot_possibly_by_enemy
        self.shot_possibly_at_enemy = shot_possibly_at_enemy
        self.pos_self = pos_self
        self.pos_enemy = pos_enemy
        self.angle_self = angle_self
        self.angle_enemy = angle_enemy
        self.map_size = map_size
        self.self_faces_enemy = Action.angleTo(self.pos_self, self.pos_enemy, self.angle_self)
        self.enemy_faces_self = Action.angleTo(self.pos_enemy, self.pos_self, self.angle_enemy)
        self.hash = None
        #m = hashlib.md5()
        #m.update(json.dumps(key))
        #int(m.hexdigest(), 16)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if self.hash is not None and other.hash is not None:
                return self.hash == other.hash

        key_self = (str(self.energy_self) +
               str(self.energy_enemy) +
               str(self.pos_self[0]) +
               str(self.pos_self[1]) +
               str(self.pos_enemy[0]) +
               str(self.pos_enemy[1]) +
               str(self.angle_self) +
               str(self.angle_enemy))
        key_other = (str(self.energy_self) +
               str(other.energy_enemy) +
               str(other.pos_self[0]) +
               str(other.pos_self[1]) +
               str(other.pos_enemy[0]) +
               str(other.pos_enemy[1]) +
               str(other.angle_self) +
               str(other.angle_enemy))
        return key_self == key_other

    def __hash__(self):
        if self.hash is None:
            key = (str(self.energy_self) +
                   str(self.energy_enemy) +
                   str(self.pos_self[0]) +
                   str(self.pos_self[1]) +
                   str(self.pos_enemy[0]) +
                   str(self.pos_enemy[1]) +
                   str(self.angle_self) +
                   str(self.angle_enemy))
            self.hash = hash(key)
        return self.hash

    """
    def __eq__(self, other):
        if isinstance(other, State):
            if self.energy_self == other.__energy_self \
                and self.energy_enemy == other.__energy_enemy \
                and self.shot_possibly_by_enemy == other.__shot_possibly_by_enemy \
                and self.shot_possibly_at_enemy == other.__shot_possibly_at_enemy \
                and self.pos_self == other.__pos_self \
                and self.pos_enemy == other.__pos_enemy \
                and self.angle_self == other.__angle_self \
                and self.angle_enemy == other.__angle_enemy \
                and self.map_size == other.__map_size \
                and self.self_faces_enemy == other.__self_faces_enemy \
                and self.enemy_faces_self == other.__enemy_faces_self:
                return True
            else:
                return False
        # Error Handling
        errString = "".join(["unsupported operand type(s) for +: 'State' and '", type(other).__name__, "'"])
        raise NameError(errString)
    """

    """
    def eval(self):
        utility = 5*(-self.energy_enemy + self.energy_self)

        if self.energy_enemy <= 0:
            utility += 1000
        if self.energy_self <= 0:
            utility -= 1000

        angle_to_enemy = Action.angleTo(self.pos_self, self.pos_enemy, self.angle_self)
        self.self_faces_enemy = angle_to_enemy
        utility -= angle_to_enemy
        angle_to_self = Action.angleTo(self.pos_enemy, self.pos_self, self.angle_enemy)
        self.enemy_faces_self = angle_to_self
        utility += angle_to_self

        if self.shot_possibly_by_enemy:
            utility -= 0
        if self.shot_possibly_at_enemy:
            utility += 0
        return self, utility
    """

    def apply_action(self, enemy,  action):
        if action == "turn_right":
           return self.apply_turn_right(enemy)
        elif action == "turn_left":
            return self.apply_turn_left(enemy)
        elif action == "forward":
            return self.apply_forward(enemy)
        elif action == "backward":
            return self.apply_backward(enemy)
        elif action == "shoot":
            return self.apply_shoot(enemy)

    def apply_turn_right(self, enemy):
        new_angle_enemy = self.angle_enemy
        new_angle_self = self.angle_self
        new_shot_possibly_by_enemy = self.shot_possibly_by_enemy
        new_shot_possibly_at_enemy = self.shot_possibly_at_enemy
        if enemy:
            new_angle_enemy = self.angle_enemy + Action.get_turn_right()
            new_shot_possibly_by_enemy = Action.faces(self.pos_enemy, self.pos_self, new_angle_enemy)
        else:
            new_angle_self = self.angle_self + Action.get_turn_right()
            new_shot_possibly_at_enemy = Action.faces(self.pos_self, self.pos_enemy, new_angle_self)

        new_state = State(self.energy_self, self.energy_enemy, new_shot_possibly_by_enemy, new_shot_possibly_at_enemy,
                          self.pos_self, self.pos_enemy, new_angle_self, new_angle_enemy, self.map_size)
        return new_state

    def apply_turn_left(self, enemy):
        new_angle_enemy = self.angle_enemy
        new_angle_self = self.angle_self
        new_shot_possibly_by_enemy = self.shot_possibly_by_enemy
        new_shot_possibly_at_enemy = self.shot_possibly_at_enemy
        if enemy:
            new_angle_enemy = self.angle_enemy + Action.get_turn_left()
            new_shot_possibly_by_enemy = Action.faces(self.pos_enemy, self.pos_self, new_angle_enemy)
        else:
            new_angle_self = self.angle_self + Action.get_turn_left()
            new_shot_possibly_at_enemy = Action.faces(self.pos_self, self.pos_enemy, new_angle_self)

        new_state = State(self.energy_self, self.energy_enemy, new_shot_possibly_by_enemy, new_shot_possibly_at_enemy,
                          self.pos_self, self.pos_enemy, new_angle_self, new_angle_enemy, self.map_size)
        return new_state

    def apply_forward(self, enemy):
        new_pos_enemy = [self.pos_enemy[0], self.pos_enemy[1]]
        new_pos_self = [self.pos_self[0], self.pos_self[1]]
        new_shot_possibly_by_enemy = self.shot_possibly_by_enemy
        new_shot_possibly_at_enemy = self.shot_possibly_at_enemy
        if enemy:
            new_pos_enemy = self.move(self.pos_enemy, Action.get_forward(), self.angle_enemy)
            new_shot_possibly_by_enemy = Action.faces(new_pos_enemy, self.pos_self, self.angle_enemy)
            new_shot_possibly_at_enemy = Action.faces(self.pos_self, new_pos_enemy, self.angle_self)
        else:
            new_pos_self = self.move(self.pos_self, Action.get_forward(), self.angle_self)
            new_shot_possibly_at_enemy = Action.faces(new_pos_self, self.pos_enemy, self.angle_self)
            new_shot_possibly_by_enemy = Action.faces(self.pos_enemy, new_pos_self, self.angle_enemy)

        new_state = State(self.energy_self, self.energy_enemy, new_shot_possibly_by_enemy, new_shot_possibly_at_enemy,
                          new_pos_self, new_pos_enemy, self.angle_self, self.angle_enemy, self.map_size)
        return new_state

    def apply_backward(self, enemy):
        new_pos_enemy = [self.pos_enemy[0], self.pos_enemy[1]]
        new_pos_self = [self.pos_self[0], self.pos_self[1]]
        new_shot_possibly_by_enemy = self.shot_possibly_by_enemy
        new_shot_possibly_at_enemy = self.shot_possibly_at_enemy
        if enemy:
            new_pos_enemy = self.move(self.pos_enemy, Action.get_backward(), self.angle_enemy)
            new_shot_possibly_by_enemy = Action.faces(new_pos_enemy, self.pos_self, self.angle_enemy)
            new_shot_possibly_at_enemy = Action.faces(self.pos_self, new_pos_enemy, self.angle_self)
        else:
            new_pos_self = self.move(self.pos_self, Action.get_backward(), self.angle_self)
            new_shot_possibly_at_enemy = Action.faces(new_pos_self, self.pos_enemy, self.angle_self)
            new_shot_possibly_by_enemy = Action.faces(self.pos_enemy, new_pos_self, self.angle_enemy)

        new_state = State(self.energy_self, self.energy_enemy, new_shot_possibly_by_enemy, new_shot_possibly_at_enemy,
                          new_pos_self, new_pos_enemy, self.angle_self, self.angle_enemy, self.map_size)
        return new_state

    def apply_shoot(self, enemy):
        new_energy_self = self.energy_self
        new_energy_enemy = self.energy_enemy
        if enemy:
            if self.shot_possibly_by_enemy:
                new_energy_self = self.energy_self - 3*Action.get_shoot()
                new_energy_enemy = self.energy_enemy + 2*Action.get_shoot()
            else:
                new_energy_enemy = self.energy_enemy - Action.get_shoot()

        else:
            if self.shot_possibly_at_enemy:
                new_energy_enemy = self.energy_enemy - 3*Action.get_shoot()
                new_energy_self = self.energy_self + 2*Action.get_shoot()
            else:
                new_energy_self = self.energy_self - Action.get_shoot()

        new_state = State(new_energy_self, new_energy_enemy, self.shot_possibly_by_enemy, self.shot_possibly_at_enemy,
                          self.pos_self, self.pos_enemy, self.angle_self, self.angle_enemy, self.map_size)

        return new_state

    def move(self, pos, value, angle):
        newpos = [0, 0]
        dx = - math.sin(math.radians(angle-90))*value
        dy = math.cos(math.radians(angle-90))*value
        newpos[0] = pos[0] + dx
        newpos[1] = pos[1] + dy
        return newpos

    def is_terminal(self):
        if self.energy_enemy <=0 or self.energy_self <= 0:
            return True
        else:
            return False

    def get_possible_actions(self, enemy):
        forward = False
        backward = False
        pos = self.pos_self
        angle = self.angle_self
        if enemy:
            pos = self.pos_enemy
            angle = self.angle_enemy

        newpos_self = self.move(pos, Action.get_forward(), angle)
        if 50 <= newpos_self[0] <= self.map_size[0]-50 and 50 <= newpos_self[1] <= self.map_size[1] - 50:
            forward = True
        newpos_self = self.move(pos, Action.get_backward(), angle)
        if 50 <= newpos_self[0] <= self.map_size[0] - 50 and 50 <= newpos_self[1] <= self.map_size[1] - 50:
            backward= True

        new_actions = {}
        for action in Action.get_actions():
            if action == "forward" and not forward or action == "backward" and not backward:
                pass
            else:
                new_actions[action] = Action.get_actions()[action]
        return new_actions
