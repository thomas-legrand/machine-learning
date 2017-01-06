import math
import numpy as np
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression

np.random.seed(0)


def compute_gradient_component(X, y, j, w):
    n = len(y)
    num = X[:, j] * y
    den = [1 + math.exp(w[j] * z) for z in num]
    grad_comp = - (float(1) / n) * sum(num / den)
    return grad_comp


def fit_logistic_regression(X, y, learning_rate, eps):
    # initialize weights to 0 and gradient to 1
    m = X.shape[1]
    w = np.array([0] * m)
    norm_grad = 1
    while norm_grad > eps:
        # compute the gradient, component by component
        grad = [compute_gradient_component(X, y, j, w) for j in range(m)]
        # compute L2 norm of the gradient
        norm_grad = np.linalg.norm(grad, 2)
        # update the weigths
        w = [w[j] - learning_rate * grad[j] for j in range(m)]
    return np.array(w)


def prediction(X_test, w):
    y_pred = [1 / (1 + math.exp(-z)) for z in X_test.dot(w)]
    return np.array(y_pred)


def fit_and_evaluate(X_train, y_train, X_test, y_test, learning_rate, eps):
    w = fit_logistic_regression(X_train, y_train, learning_rate, eps)
    y_prob = prediction(X_test, w)
    y_pred = np.array([0] * len(y_prob))
    y_pred[y_prob > .5] = 1
    y_pred[y_prob <= .5] = -1
    return metrics.confusion_matrix(y_test, y_pred)


def fit_and_evaluate_benchmark(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return metrics.confusion_matrix(y_test, y_pred)


def main():
    X, y = datasets.make_classification(n_samples=100, n_features=5, n_informative=2)
    # for the logistic regression trick to work, convert 0 labels to -1
    # logistic regression trick: P(y|x) = theta(y * t(w) * x) where theta is the logistic function
    y[y == 0] = -1
    train_samples = 50
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]
    print(fit_and_evaluate(X_train, y_train, X_test, y_test, .1, 1e-6))
    print(fit_and_evaluate_benchmark(X_train, y_train, X_test, y_test))


if __name__ == '__main__':
    main()
