from Regression import RegressionGradient
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')
def h(x):
    return np.sin(x) + 0.3 * x - 1

dn = np.random.uniform(-5, 5, 15).reshape((-1 ,1))
dn = np.sort(dn, axis=0)
dn_labels = h(dn)

dtest = np.random.uniform(-5, 5, 100).reshape((-1,1))
dtest= np.sort(dtest, axis=0)
dtest_labels = h(dtest)


def phi(X, l=1):
    phi = []
    for x in X:
        powers = []
        for n in x:
            powers.append([n ** power for power in range(1, l + 1)])
        powers = np.array(powers)

        phi_row = []
        for i in range(powers.shape[0]):
            current_row = powers[i]
            phi_row.extend(current_row)
            for j in range(i + 1, powers.shape[0]):
                second_row = powers[j]
                for k in range(l):
                    for m in range(l - k - 1):
                        phi_row.append(current_row[k] * second_row[m])
        phi.append(phi_row)
    return np.array(phi)


powers = [20]


def min_max_normalization(dn, high=1.0, low=-1.0):
    mins = np.min(dn, axis=0)
    maxs = np.max(dn, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - dn)) / rng)


legend = []
for power in powers:
    phi_dn = phi(dn, power)
    phi_dn = min_max_normalization(phi_dn)
    phi_dtest = phi(dtest, power)
    phi_dtest = min_max_normalization(phi_dtest)
    regression_gradient = RegressionGradient(step_size=0.01, lambda_l2=0.01, epochs=100000)
    regression_gradient.train(phi_dn, dn_labels)
    predictions = regression_gradient.predict(phi_dtest)
    plt.plot(dtest, predictions)
    legend.append("l={}".format(power))

plt.scatter(dtest, dtest_labels)
legend.append("dtest")
plt.legend(legend)
plt.title("Prédictions sur Dtest avec ")
plt.show()
