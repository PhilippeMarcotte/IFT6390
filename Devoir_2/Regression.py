import numpy as np
import math
class RegressionGradient:
    def __init__(self, step_size=0.5, lambda_l2=1, epochs=100):
        self.step_size = step_size
        self.lambda_l2 = lambda_l2
        self.epochs = epochs

    def compute_cost(self, error):
        return ((error.T.dot(error) + self.lambda_l2 * self.theta.T.dot(self.theta))).squeeze()

    def gradient_step(self):
        error = self.train_inputs.dot(self.theta).reshape((-1,1)) - self.train_targets
        cost = self.compute_cost(error)
        gradient = self.train_inputs.T.dot(error)

        self.theta = self.theta - self.step_size / self.train_size * (gradient + self.lambda_l2 * self.theta)
        return cost

    def train(self, train_inputs, train_targets):
        #self.train_inputs = np.vstack((np.ones(train_inputs.shape[0]), train_inputs)).T
        self.train_inputs = np.insert(train_inputs, 0, 1, axis=1)
        self.train_size = self.train_inputs.shape[0]

        self.train_targets = train_targets

        self.theta = np.random.uniform(size=(train_inputs.shape[1] + 1, 1)) if train_inputs.ndim > 1 else np.random.uniform(size=(2, 1))

        self.i = 0
        theta_isnan = False
        for self.i in range(self.epochs):
            cost = self.gradient_step()
            theta_isnan = np.any(np.isnan(self.theta))

        print("Iteration : {}, cost : {}".format(self.i + 1, cost))

        return self.theta

    def predict(self, test_inputs):
        test_inputs = np.insert(test_inputs, 0, 1, axis=1)
        return test_inputs.dot(self.theta).reshape((-1, 1))
