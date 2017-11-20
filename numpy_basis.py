import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    """
    Compute the gradient of the sigmoid function w.r.t x

    Arguments:
    x -- A scalar or numpy array

    Return:
    dx -- Computed gradient
    """

    s = sigmoid(x)
    dx = s * (1 - s)

    return dx


def normalizeRows(x):
    """
    Implement a function to normalizes each row of the matrix x (to have unit length)

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix
    """

    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm

    return x


def softmax(x):
    """
    Calculates the softmax for each row of the input x

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    s -- A numpy matrix equal to the softmax of x
    """

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum

    return s
