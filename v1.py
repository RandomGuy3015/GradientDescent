
from dataclasses import dataclass


@dataclass
class Vec:
    x1: float
    x2: float

    def __mul__(self, other):  # this implements the dot product
        if isinstance(other, Vec):
            return self.norm().x1 * other.x1 + self.norm().x2 + other.x2
        else:
            # this implements constant multiplication
            return Vec(self.x1 * other, self.x2 * other)

    def norm(self):  # this normalizes the vector
        factor = (self.x1 ** 2 + self.x2 ** 2) ** 0.5
        return Vec(self.x1 / factor, self.x2 / factor)

    def __add__(self, other):
        return Vec(self.x1 + other.x1, self.x2 + other.x2)

    def __neg__(self):
        return Vec(-self.x1, -self.x2)


def f(v):
    return 2 * v.x1 ** 2 + 2 * v.x2 ** 2 + 3 * v.x1 * v.x2 + v.x2 - v.x1 + v.x2


def grad(v):
    return Vec(4 * v.x1 + 3 * v.x2 - 1, 4 * v.x2 + 3 * v.x1 + 1)


def arminjo(xk, dk, beta, delta):
    tk = 1
    while f(xk + dk * tk) - f(xk) > grad(xk) * dk * tk * delta:
        tk = tk * beta
    return tk


def minimier():
    x1_0 = 1
    x2_0 = 1
    xk = Vec(x1_0, x2_0)
    delta = 0.0001
    beta = 0.9

    for k in range(5):
        # richtung bestimmen
        dk = -grad(xk)

        # schrittweite
        tk = arminjo(xk, dk, beta, delta)

        # prints\
        print("k: ", k, "xk: ", xk, "grad: ", dk, "schritt: ", tk, "naechster schritt: ", dk * tk)

        # naechstes xk+1
        xk = xk + dk * tk


minimier()
