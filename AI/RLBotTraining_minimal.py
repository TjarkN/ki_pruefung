from AI.game_environment import GameEnvironment
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from AI.actions import Action
import numpy as np
from AI.state import State

class Training:

    def __init__(self, x, y, botList, no_graphics=True):
        # parameters

        self.num_actions = len(Action.get_actions())
        self.width = x
        self.height = y
        self.botList = botList
        self.num_state_vars = 3
        self.epochs = 100
        self.model = self.create_model()
        self.exp_replay = ExperienceReplay()
        self.env = GameEnvironment(self.width, self.height, no_grafics=no_graphics)
        self.reset_game(True)


    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_RLBotTjark.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_RLBotTjark.h5")

    def training_done(self):
        print("Training done")
        self.save_model()
        print("Model saved")

    def train(self, state, action, new_state, reward, game_over):

        # Implement the Training here
        for e in range(self.epochs):
            loss = 0.0
            # Experience Replay
            # store experience
            self.exp_replay.remember([state, action, reward, new_state], game_over)
            # load batch of experiences
            inputs, targets = self.exp_replay.get_batch(self.model, batch_size=10)
            # train model on experience
            batch_loss = self.model.train_on_batch(inputs, targets)
            loss += batch_loss
            print("Epoch {:03d}/{:03d} | Loss {:.4f} ".format(e, self.epochs, loss))

        self.training_done()
        return

    def reset_game(self, first_time=False):
        botList = []
        models = []
        trainings = []
        for bot in self.botList:
            botList.append(bot[0])
            if bot[1]:
                models.append(self.model)
                trainings.append(self)
            else:
                models.append(None)
                trainings.append(None)

        if first_time:
            self.env.start(botList, models, trainings)
        else:
            self.env.restart(botList, models, trainings)

    def create_model(self):
        # You have to set up the model with keras here
        model = Sequential()
        model.add(Dense(20, input_shape=(self.num_state_vars,),kernel_initializer="normal",
                        kernel_regularizer=regularizers.l1(0.00001), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(optimizer='nadam', loss='mean_squared_error')
        #history = model.fit(X, Y, epochs=1000, verbose=False)
        return model


class ExperienceReplay(object):

    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions do not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i:i + 1] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t, verbose=0)[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1, verbose=0)[0])

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets