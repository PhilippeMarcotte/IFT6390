import numpy as np

class RegressionGradient:
    def __init__(self, step_size=0.5, lambda_l2=1, nb_steps=1000):
        self.step_size = step_size
        self.lambda_l2 = lambda_l2
        self.nb_steps = nb_steps

    def compute_cost(self, error):
        return error.T.dot(error) + self.lambda_l2 * self.theta.T.dot(self.theta)

    def gradient_step(self):
        error = self.train_inputs.dot(self.theta) - self.train_targets
        cost = self.compute_cost(error)
        gradient = 2 * self.train_inputs.T.dot(error) / self.train_size

        self.theta = self.theta - self.step_size * gradient + 2 * self.lambda_l2 * self.theta
        return cost

    def train(self, train_inputs, train_targets):
        self.train_inputs = np.vstack((np.ones(train_inputs.shape[0]), train_inputs)).T
        self.train_size = self.train_inputs.shape[0]

        self.train_targets = train_targets

        self.theta = np.random.uniform(size=(train_inputs.shape[1] + 1)) if train_inputs.ndim > 1 else np.random.uniform(size=(2))

        for i in range(self.nb_steps):
            cost = self.gradient_step()
            print("Iteration : {}, cost : {}".format(i, cost))

        return self.theta

    def predict(self, test_inputs):
        test_inputs = np.vstack((np.ones(test_inputs.shape[0]), test_inputs)).T
        return test_inputs.dot(self.theta)
