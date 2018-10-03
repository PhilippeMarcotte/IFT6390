from Regression import RegressionGradient
import matplotlib.pyplot as plt
import numpy as np

def h(x):
    return np.sin(x) + 0.3 * x - 1

np.random.seed(100)
X = np.random.uniform(-5, 5, 15)
Y = h(X)

regression_gradient = RegressionGradient(step_size=0.01, lambda_l2=0)
regression_gradient_medLambda = RegressionGradient(step_size=0.001, lambda_l2=0.0001)
#regression_gradient_highLambda = RegressionGradient(step_size=0.001, lambda_l2=0.1)

regression_gradient.train(X, Y)
regression_gradient_medLambda.train(X, Y)
#regression_gradient_highLambda.train(X, Y)

xgrid = np.arange(-10, 10)

figure = plt.figure()
ax = figure.add_subplot(1,1,1)
ax.set_xticks(xgrid)
ax.scatter(X, Y)
ax.plot(X, regression_gradient.predict(X), color='orange')
ax.plot(X, regression_gradient_medLambda.predict(X), color='red')
#ax.plot(X, regression_gradient_highLambda.predict(X), color='green')
ax.legend(["h(x)", "regression_gradient (lambda=0)", "regression_gradient (lambda=0.5)"])
plt.show()