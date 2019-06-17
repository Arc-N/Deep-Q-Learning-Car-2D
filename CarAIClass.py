import os
import tensorflow as tf
from keras.layers import Dense
import random
from keras.models import Sequential
import numpy as np
from keras.optimizers import Adam
from collections import deque
from keras.models import load_model


class CarAI:
    def __init__(self, sensors=6, moves=2, gamma=0.95, alpha=0.003, epsilon=1.0, epsilonEnd=0.01):
        self.sensors = sensors
        self.moves = moves

        self.memory = deque(maxlen=1000)  # used to do batch learning
        # hyperparamteres
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilonEnd = epsilonEnd

        self.alpha = alpha
        #self.model = load_model("./SavedModel/preTrained.hdf5")
        self.model = self._model()
    def _model(self):
        # standard keras model for fully connected neural net
        model = Sequential()
        model.add(Dense(36, input_dim=self.sensors, activation='relu'))
        model.add(Dense(36, activation='relu'))
        # we don't want to actually do any activation on our output since we are looking for the actual values
        model.add(Dense(self.moves, activation='linear'))
        # I am using MSELoss since it's reliable, but there could be better options.
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        # Adam seems to work fine, other optimizers could work better.
        return model
    def getMove(self, sensors):
        # get either greedy move or random move
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.moves)  # return a random move
        moves = self.model.predict(sensors)
        return np.argmax(moves[0])  # return the greedy move
    def store(self, sensors, move, reward, sensorsNext, done):
        # store each state
        self.memory.append((sensors, move, reward, sensorsNext, done))
    def learn(self, learnSize):
        miniMem = random.sample(self.memory, learnSize)
        for sensors, move, reward, sensorsNext, done in miniMem:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(sensorsNext)[0]))
            target_f = self.model.predict(sensors)
            target_f[0][move] = target

            self.model.fit(sensors, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilonEnd:
            self.epsilon *= 0.97
