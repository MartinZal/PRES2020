from heuristic_optimization.optimizers import ParticleSwarmOptimizer
import numpy as np
import copy
from heat import fun
from heat.fun import heatIn, qprl, qp, ql, qprp, moveFun1, neighFun, heatInAll, heatPoint, heat, yout
import pygmo as pg
import time
import matplotlib.pyplot as plt
# PREPROCESSING

l = 0.1
h = 1
d = 1
qh = 0
qd = 0
qf = 0
qb = 0
rho = 760
rhov = 1.1614
lambd = 0.2
T0p = 40
Nx = 8
Ny = 6
Nz = 6
Nzv = 7
dt = 0.1
tmax = 1800
qloss = 0
e = 0.03
cp = 1007
mi = 184.6 * 10 ** (-7)
k = 0.0272
cv = 1007
kinv = 15.52 * 10 ** (-6)

dz = d / (Nz - 1)
v = 1.7

# predpocet intevalu
a = np.zeros([2, Nz])
a[0, 0] = 0
a[1, -1] = d
for i in range(0, Nz - 1):
    a[1, i] = i * dz + dz / 2
    a[0, i + 1] = a[1, i]

dzv = d / (Nzv - 1)

b = np.zeros([2, Nzv])
b[0, 0] = 0
b[1, -1] = d

for i in range(0, Nzv - 1):
    b[1, i] = i * dzv + dzv / 2
    b[0, i + 1] = b[1, i]

neighField = fun.neighFun(a, b, Nz, Nzv)


def heat1(x):
    heatSum = 0
    dzp = dt * v
    dz = d / (Nz - 1)
    if (dzp >= dz):
        print('Neni splnena podminka na rychlost, muze dojit k preskoceni jednoho KO')

    alpha = fun.calcDN(h, e, cv, mi, k, v, kinv)

    def c(T):
        return 2000 + 43770 * np.exp(-(T - 43) ** 2 / 4.8)

    ctv = 0
    dx = x[0] / (Nx - 1)
    dy = h / (Ny - 1)
    # x[2] = 10 / (x[0] * x[1] * rho)
    dz = d / (Nz - 1)

    T = np.zeros([Nx, Ny, Nz, int(tmax / dt) + 1])
    Tvz = np.zeros([Ny, 15000, int(tmax / dt) + 2, 2])

    T[:, :, :, 0].fill(T0p)
    Tvz[:, :, :, :].fill(18)

    bIt = copy.copy(b)
    neighField = neighFun(a, b, Nz, Nzv)

    for p in np.arange(0, int(tmax / dt), 1):

        # vzduch leva rohy

        Tvz[0, 0, p + 1, 0] = Tvz[0, 0, p, 0] + (2 * dt) / (e * dy * neighField[0, 0] * cv * rhov) * \
        (qprl(0, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, 0, p + 1, 0] = Tvz[-1, 0, p, 0] + (2 * dt) / (e * dy * neighField[0, 0] * cv * rhov) * \
        (qprl(-1, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[0, Nzv - 1, p + 1, 0] = Tvz[0, Nzv - 1, p, 0] + (2 * dt) / (e * dy * neighField[-1, -1] * cv * rhov) * \
        (qprl(0, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, Nzv - 1, p + 1, 0] = Tvz[-1, Nzv - 1, p, 0] + (2 * dt) / (e * dy * neighField[-1, -1] * cv * rhov) * \
        (qprl(-1, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        # vzduch prava rohy

        Tvz[0, 0, p + 1, 1] = Tvz[0, 0, p, 1] + (2 * dt) / (e * dy * neighField[0, 0] * cv * rhov) * \
        (qprp(0, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, 0, p + 1, 1] = Tvz[-1, 0, p, 1] + (2 * dt) / (e * dy * neighField[0, 0] * cv * rhov) * \
        (qprp(-1, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[0, Nzv - 1, p + 1, 1] = Tvz[0, Nzv - 1, p, 1] + (2 * dt) / (e * dy * neighField[-1, -1] * cv * rhov) * \
        (qprp(0, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, Nzv - 1, p + 1, 1] = Tvz[-1, Nzv - 1, p, 1] + (2 * dt) / (e * dy * neighField[-1, -1] * cv * rhov) * \
        (qprp(-1, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        T[0, 0, 0, p + 1] = T[0, 0, 0, p] + (2 * dt) / (c(T[0, 0, 0, p]) * rho) * \
        (ql(0, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[0, 0, 0, p]) * rho) * \
        ((T[1, 0, 0, p] - T[0, 0, 0, p]) / dx ** 2 + (T[0, 1, 0, p] - T[0, 0, 0, p]) / dy ** 2 +
        (T[0, 0, 1, p] - T[0, 0, 0, p]) / dz ** 2)

        T[-1, 0, 0, p + 1] = T[-1, 0, 0, p] + (2 * dt) / (c(T[-1, 0, 0, p]) * rho) * \
        (qp(0, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[-1, 0, 0, p]) * rho) * ((T[-2, 0, 0, p] -
        T[-1, 0, 0, p]) / dx ** 2 + (T[-1, 1, 0, p] - T[-1, 0, 0, p]) / dy ** 2 +
        (T[-1, 0, 1, p] - T[-1, 0, 0, p]) / dz ** 2)

        T[0, -1, 0, p + 1] = T[0, -1, 0, p] + (2 * dt) / (c(T[0, -1, 0, p]) * rho) * \
        (ql(-1, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[0, -1, 0, p]) * rho) * ((T[1, -1, 0, p] - T[0, -1, 0, p]) / dx ** 2 +
        (T[0, -2, 0, p] - T[0, -1, 0, p]) / dy ** 2 + (T[0, -1, 1, p] - T[0, -1, 0, p]) / dz ** 2)

        T[-1, -1, 0, p + 1] = T[-1, -1, 0, p] + (2 * dt) / (c(T[-1, -1, 0, p]) * rho) * (
        qp(-1, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[-1, -1, 0, p]) * rho) * ((T[-2, -1, 0, p] - T[-1, -1, 0, p]) / dx ** 2 +
        (T[-1, -2, 0, p] - T[-1, -1, 0, p]) / dy ** 2 + (T[-1, -1, 1, p] - T[-1, -1, 0, p]) / dz ** 2)

        T[0, 0, -1, p + 1] = T[0, 0, -1, p] + (2 * dt) / (c(T[0, 0, -1, p]) * rho) * \
        (ql(0, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[0, 0, -1, p]) * rho) * (
        (T[1, 0, -1, p] - T[0, 0, -1, p]) / dx ** 2 +
        (T[0, 1, -1, p] - T[0, 0, -1, p]) / dy ** 2 + (
        T[0, 0, -2, p] - T[0, 0, -1, p]) / dz ** 2)

        T[-1, 0, -1, p + 1] = T[-1, 0, -1, p] + (2 * dt) / (c(T[-1, 0, -1, p]) * rho) * \
        (qp(0, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[-1, 0, -1, p]) * rho) * \
        ((T[-2, 0, -1, p] - T[-1, 0, -1, p]) / dx ** 2 +
        (T[-1, 1, -1, p] - T[-1, 0, -1, p]) / dy ** 2 + (T[-1, 0, -2, p] - T[-1, 0, -1, p]) / dz ** 2)

        T[0, -1, -1, p + 1] = T[0, -1, -1, p] + (2 * dt) / (c(T[0, -1, -1, p]) * rho) * \
        (ql(-1, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[0, -1, -1, p]) * rho) * ((T[1, -1, -1, p] - T[0, -1, -1, p]) / dx ** 2 +
        (T[0, -2, -1, p] - T[0, -1, -1, p]) / dy ** 2 + (T[0, -1, -2, p] - T[0, -1, -1, p]) / dz ** 2)

        T[-1, -1, -1, p + 1] = T[-1, -1, -1, p] + (2 * dt) / (c(T[-1, -1, -1, p]) * rho) * (
        qp(-1, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[-1, -1, -1, p]) * rho) * ((T[-2, -1, -1, p] -
        T[-1, -1, -1, p]) / dx ** 2 + (T[-1, -2, -1, p] - T[-1, -1, -1, p]) / dy ** 2 +
        (T[-1, -1, -2, p] - T[-1, -1, -1, p]) / dz ** 2)

        for m in range(1, Nx - 1):

            T[m, 0, 0, p + 1] = T[m, 0, 0, p] + (2 * dt) / (c(T[m, 0, 0, p]) * rho) * (qh / dy + qf / dz) + \
            (2 * lambd * dt) / (c(T[m, 0, 0, p]) * rho) * ((T[m - 1, 0, 0, p] + T[m + 1, 0, 0, p] -
            2 * T[m, 0, 0, p]) / (dx ** 2 * 2) + (T[m, 1, 0, p] - T[m, 0, 0, p]) / dy ** 2 +
            (T[m, 0, 1, p] - T[m, 0, 0, p]) / dz ** 2)

            T[m, -1, 0, p + 1] = T[m, -1, 0, p] + (2 * dt) / (c(T[m, -1, 0, p]) * rho) * (qd / dy + qf / dz) + \
            (2 * lambd * dt) / (c(T[m, -1, 0, p]) * rho) * ((T[m - 1, -1, 0, p] + T[m + 1, -1, 0, p] -
            2 * T[m, -1, 0, p]) / (dx ** 2 * 2) + (T[m, -2, 0, p] - T[m, -1, 0, p]) / dy ** 2 +
            (T[m, -1, 1, p] - T[m, -1, 0, p]) / dz ** 2)

            T[m, 0, -1, p + 1] = T[m, 0, -1, p] + (2 * dt) / (c(T[m, 0, -1, p]) * rho) * (qh / dy + qb / dz) +\
            (2 * lambd * dt) / (c(T[m, 0, -1, p]) * rho) * ((T[m - 1, 0, -1, p] +T[m + 1, 0, -1, p] -
            2 *T[m, 0, -1, p]) / (dx ** 2 * 2) + (T[m, 1, -1, p] -T[m, 0, -1, p]) / dy ** 2 + (T[m, 0, -2, p] -
            T[m, 0, -1, p]) / dz ** 2)

            T[m, -1, -1, p + 1] = T[m, -1, -1, p] + (2 * dt) / (c(T[m, -1, -1, p]) * rho) * (qd / dy + qb / dz) + \
            (2 * lambd * dt) / (c(T[m, -1, -1, p]) * rho) * ((T[m - 1, -1, -1, p] + T[m + 1, -1, -1, p] -
            2 *T[m, -1, -1, p]) / (dx ** 2 * 2) + (T[m, -2, -1, p] - T[m, -1, -1, p]) / dy ** 2 +
            (T[m, -1, -2, p] - T[m, -1, -1, p]) / dz ** 2)

            for n in range(1, Ny - 1):

                T[0, n, 0, p + 1] = T[0, n, 0, p] + (2 * dt) / (c(T[0, n, 0, p]) * rho) * \
                (ql(n, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qf / dz) + \
                (2 * lambd * dt) / (c(T[0, n, 0, p]) * rho) * ((T[0, n + 1, 0, p] +
                T[0, n - 1, 0, p] - 2 * T[0, n, 0, p]) / (dy ** 2 * 2) + (T[1, n, 0, p] -
                T[0, n, 0, p]) / dx ** 2 + (T[0, n, 1, p] - T[0, n, 0, p]) / dz ** 2)

                T[-1, n, 0, p + 1] = T[-1, n, 0, p] + (2 * dt) / (c(T[-1, n, 0, p]) * rho) * (
                qp(n, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qf / dz) + \
                (2 * lambd * dt) / (c(T[-1, n, 0, p]) * rho) * ((T[-1, n + 1, 0, p] +
                T[-1, n - 1, 0, p] - 2 * T[-1, n, 0, p]) / (dy ** 2 * 2) + (T[-2, n, 0, p] -
                T[-1, n, 0, p]) / dx ** 2 + (T[-1, n, 1, p] - T[-1, n, 0, p]) / dz ** 2)

                T[0, n, -1, p + 1] = T[0, n, -1, p] + (2 * dt) / (c(T[0, n, -1, p]) * rho) * (
                ql(n, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qb / dz) + (2 * lambd * dt) / (
                c(T[0, n, -1, p]) * rho) * ((T[0, n + 1, -1, p] + T[0, n - 1, -1, p] -
                2 * T[0, n, -1, p]) / (dy ** 2 * 2) + (T[1, n, -1, p] - T[0, n, -1, p]) / dx ** 2 +
                (T[0, n, -2, p] - T[0, n, -1, p]) / dz ** 2)

                T[-1, n, -1, p + 1] = T[-1, n, -1, p] + (2 * dt) / (c(T[-1, n, -1, p]) * rho) * (
                qp(n, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qb / dz) + \
                (2 * lambd * dt) / (c(T[-1, n, -1, p]) * rho) * ((T[-1, n + 1, -1, p] + T[-1, n - 1, -1, p] -
                2 *T[-1, n, -1, p]) / (dy ** 2 * 2) + (T[-2, n, -1, p] - T[-1, n, -1, p]) / dx ** 2 +
                (T[-1, n, -2, p] - T[-1, n, -1, p]) / dz ** 2)

                T[m, n, 0, p + 1] = T[m, n, 0, p] + (2 * qf * dt) / (rho * c(T[m, n, 0, p]) * dz) + (2 * lambd * dt) / (
                c(T[m, n, 0, p]) * rho) * ((T[m + 1, n, 0, p] + T[m - 1, n, 0, p] - 2 * T[m, n, 0, p]) / (
                dx ** 2 * 2) + (T[m, n + 1, 0, p] + T[m, n - 1, 0, p] - 2 * T[m, n, 0, p]) / (dy ** 2 * 2) +
                (T[m, n, 1, p] - T[m, n, 0, p]) / dz ** 2)

                T[m, n, -1, p + 1] = T[m, n, -1, p] + (2 * qb * dt) / (rho * c(T[m, n, -1, p]) * dz) +\
                (2 * lambd * dt) / (c(T[m, n, -1, p]) * rho) * ((T[m + 1, n, -1, p] + T[m - 1, n, -1, p] -
                2 *T[m, n, -1, p]) / (dx ** 2 * 2) + (T[m, n + 1, -1, p] + T[m, n - 1, -1, p] -
                2 *T[m, n, -1, p]) / (dy ** 2 * 2) + (T[m, n, -2, p] - T[m, n, -1, p]) / dz ** 2)

                for o in range(1, Nz - 1):
                    T[m, 0, o, p + 1] = T[m, 0, o, p] + (2 * qh * dt) / (rho * c(T[m, 0, o, p]) * dy) +\
                    (2 * lambd * dt) / (c(T[m, 0, o, p]) * rho) * ((T[m + 1, 0, o, p] + T[m - 1, 0, o, p] -
                    2 *T[m, 0, o, p]) / (dx ** 2 * 2) + (T[m, 0, o + 1, p] + T[m, 0, o - 1, p] -
                    2 * T[m, 0, o, p]) / (dz ** 2 * 2) + (T[m, 1, o, p] - T[m, 0, o, p]) / dy ** 2)

                    T[m, -1, o, p + 1] = T[m, -1, o, p] + (2 * qd * dt) / (rho * c(T[m, -1, o, p]) * dy) +\
                    (2 * lambd * dt) / (c(T[m, -1, o, p]) * rho) * ((T[m + 1, -1, o, p] + T[m - 1, -1, o, p] -
                    2 * T[m, -1, o, p]) / (dx ** 2 * 2) + (T[m, -1, o + 1, p] + T[m, -1, o - 1, p] -
                    2 * T[m, -1, o, p]) / (dz ** 2 * 2) + (T[m, -2, o, p] - T[m, -1, o, p]) / dy ** 2)

                    T[0, n, o, p + 1] = T[0, n, o, p] + (2 * ql(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) * dt)/\
                    (rho * c(T[0, n, o, p]) * dx) + (2 * lambd * dt) / (c(T[0, n, o, p]) * rho) * \
                    ((T[0, n + 1, o, p] + T[0, n - 1, o, p] - 2 *T[0, n, o, p]) / (dy ** 2 * 2) +
                    (T[0, n, o + 1, p] + T[0, n, o - 1, p] - 2 * T[0, n, o, p]) / (dz ** 2 * 2) + (T[1, n, o, p] -
                     T[0, n, o, p]) / dx ** 2)

                    T[-1, n, o, p + 1] = T[-1, n, o, p] + (2 * qp(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) * dt) / (
                    rho * c(T[-1, n, o, p]) * dx) + (2 * lambd * dt) / (c(T[-1, n, o, p]) * rho) * \
                    ((T[-1, n + 1, o, p] + T[-1, n - 1, o, p] - 2 * T[-1, n, o, p]) / (dy ** 2 * 2) +
                    (T[-1, n, o + 1, p] + T[-1, n, o - 1, p] - 2 * T[-1, n, o, p]) / (dz ** 2 * 2) +
                    (T[-2, n, o, p] - T[-1, n, o, p]) / dx ** 2)

                    T[0, 0, o, p + 1] = T[0, 0, o, p] + (2 * dt) / (c(T[0, 0, o, p]) * rho) * (
                    ql(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy) + (2 * lambd * dt) / (
                    c(T[0, 0, o, p]) * rho) * ((T[0, 0, o + 1, p] + T[0, 0, o - 1, p] - 2 * T[0, 0, o, p]) /
                    (dz ** 2 * 2) + (T[1, 0, o, p] - T[0, 0, o, p]) / dx ** 2 + (T[0, 1, o, p] -
                    T[0, 0, o, p]) / dy ** 2)

                    T[0, -1, o, p + 1] = T[0, -1, o, p] + (2 * dt) / (c(T[0, -1, o, p]) * rho) * (
                    ql(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy) + (2 * lambd * dt) / (
                    c(T[0, -1, o, p]) * rho) * ((T[0, -1, o + 1, p] + T[0, -1, o - 1, p] - 2 *T[0, -1, o, p]) /
                    (dz ** 2 * 2) + (T[1, -1, o, p] - T[0, -1, o, p]) / dx ** 2 + (T[0, -2, o, p] -
                    T[0, -1, o, p]) / dy ** 2)

                    T[-1, 0, o, p + 1] = T[-1, 0, o, p] + (2 * dt) / (c(T[-1, 0, o, p]) * rho) * (
                    qp(0, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy) + (2 * lambd * dt) / \
                    (c(T[-1, 0, o, p]) * rho) * ((T[-1, 0, o + 1, p] + T[-1, 0, o - 1, p] -
                    2 * T[-1, 0, o, p]) / (dz ** 2 * 2) + (T[-2, 0, o, p] - T[-1, 0, o, p]) / dx ** 2 +
                    (T[-1, 1, o, p] - T[-1, 0, o, p]) / dy ** 2)

                    T[-1, -1, o, p + 1] = T[-1, -1, o, p] + (2 * dt) / (c(T[-1, -1, o, p]) * rho) * (
                    qp(-1, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy) + (2 * lambd * dt) / (
                    c(T[-1, -1, o, p]) * rho) * ((T[-1, -1, o + 1, p] + T[-1, -1, o - 1, p] -
                    2 * T[-1, -1, o, p]) /(dz ** 2 * 2) + (T[-2, -1, o, p] - T[-1, -1, o, p]) / dx ** 2 +
                    (T[-1, -2, o, p] - T[-1, -1, o, p]) / dy ** 2)

                    T[m, n, o, p + 1] = T[m, n, o, p] + (lambd * dt) / (rho * dx ** 2 * c(T[m, n, o, p])) * \
                    (T[m + 1, n, o, p] - 2 * T[m, n, o, p] + T[m - 1, n, o, p]) + (lambd * dt) / \
                    (rho * dy ** 2 * c(T[m, n, o, p])) * (T[m, n + 1, o, p] - 2 * T[m, n, o, p] + T[m, n - 1, o, p]) \
                    + (lambd * dt) / (rho * dz ** 2 * c(T[m, n, o, p])) * (T[m, n, o + 1, p] -
                    2 *T[m, n, o, p] + T[m, n, o - 1, p])

        for n in range(1, Ny - 1):

            Tvz[n, 0, p + 1, 0] = Tvz[n, 0, p, 0] + dt / (e * dy * neighField[0, 0] * cv * rhov) * (
                qprl(n, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

            Tvz[n, Nzv - 1, p + 1, 0] = Tvz[n, Nzv - 1, p, 0] + dt / (e * dy * neighField[-1, -1] * cv * rhov) * (
                qprl(n, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

            Tvz[n, 0, p + 1, 1] = Tvz[n, 0, p, 1] + dt / (e * dy * neighField[0, 0] * cv * rhov) * (
                qprp(n, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

            Tvz[n, Nzv - 1, p + 1, 1] = Tvz[n, Nzv - 1, p, 1] + dt / (e * dy * neighField[-1, -1] * cv * rhov) * (
                qprp(n, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

            for q in range(1, Nzv - 1):
                Tvz[n, q, p + 1, 0] = Tvz[n, q, p, 0] + dt / (e * dy * dzv * cv * rhov) * (
                    qprl(n, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[0, q, p + 1, 0] = Tvz[0, q, p, 0] + (2 * dt) / (e * dy * dzv * cv * rhov) * (
                    qprl(0, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[-1, q, p + 1, 0] = Tvz[-1, q, p, 0] + (2 * dt) / (e * dy * dzv * cv * rhov) * (
                    qprl(-1, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[n, q, p + 1, 1] = Tvz[n, q, p, 1] + dt / (e * dy * dzv * cv * rhov) * (
                    qprp(n, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[0, q, p + 1, 1] = Tvz[0, q, p, 1] + (2 * dt) / (e * dy * dzv * cv * rhov) * (
                    qprp(0, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[-1, q, p + 1, 1] = Tvz[-1, q, p, 1] + (2 * dt) / (e * dy * dzv * cv * rhov) * (
                    qprp(-1, q, p, T, Tvz, dy, alpha, neighField) + qloss)
                # krajne vzduchu, je duvod propadu teploty na krajich, tzn. jen krajni body jsou v surfu niz - kvuli 2*

        bN, nout = moveFun1(bIt, dzv, dzp, Nzv, d)
        bIt = bN
        Tvz[:, :, p + 1] = np.roll(Tvz[:, :, p + 1], nout, axis=1)
        Tvz[:, :, p + 2] = np.roll(Tvz[:, :, p + 1], nout, axis=1)

        ctv = ctv + ((Tvz[3, Nzv - 1, p, 0] + Tvz[3, Nzv - 1, p, 1]) / 2 - 65) ** 2
        for i in range(0, Ny):
            for j in range(0, Nz):
                heatSum = heatSum + (alpha / dz * np.dot(neighField[j, :],
                                                         (Tvz[i, 0:Nzv, p, 0] - T[0, i, j, p])) + alpha / dz * np.dot(
                    neighField[j, :], (Tvz[i, 0:Nzv, p, 1] - T[-1, i, j, p]))) * yout(i, dy, Ny) * dt

    heatP = heatPoint(tmax, dt, Ny, Nz, T, Nzv, Tvz, dz, alpha, neighField)
    print('done')
    W = 0.00001
    heatSum = heatSum + 1000 * 0.91 * 0.95 * h * d * tmax

    return -heatSum * W + x[0] * d * h * rho * (1 - W)



counter = 0
list = []
class heat_f2:
    counter = 0
    def fitness(self, x):
        global counter, list
        counter = counter + 1
        start = time.time()
        f2 = heat1(x)
        end = time.time()


        print('Iteration number: ' + str(counter))
        print('Elapsed time: ' + str(end - start))
        print('Fitness: ' + str(f2))
        print('Variables: Tpch, Tpch2, c0, c1, c2, sigma1, sigma2')
        print('Current x: ' + str(x))
        print('----------------------------------------------------------------')
        list.append(f2)

        # plt.figure(num=None, figsize=(17, 7), dpi=100, facecolor='w', edgecolor='k')
        # plt.subplot(1, 2, 1)
        # plt.plot(np.arange(t_start, t_max, dt), T[5, :p_max], np.arange(t_start, t_max, dt), core[:p_max])
        # plt.xlabel('Time iteration []')
        # plt.ylabel('Temperature [degC]')
        # plt.title('Temp. evolution comparison, middle of PCM layer-experiment vs simulation')
        # plt.legend(['Simulation', 'Experiment'])
        # print(T[5, :])

        # plt.subplot(1, 2, 2)
        #
        # plt.plot(np.linspace(-50, 60, 200), ceff(np.linspace(-50, 60, 200), Tpch, Tpch2, c0, c1, c2, sigma1, sigma2))
        # plt.xlabel('Temperature [degC]')
        # plt.ylabel('Effective heat capacity [J/(kg K)]')
        # plt.title(
        #     'Opt ceff = 2000 + {:05.0f}exp(-(T - {:02.2f})^2)/{:02.2f} + {:05.0f}exp(-(T - {:02.2f})^2)/{:02.2f}'.format(
        #         x[1], x[0], x[2], x[3], x[5], x[4]))

        #plt.show()
        print(list)


        # if (f2 <= np.min(list)):
        #     plt.savefig("Images/salthydrate/{:02.0f}{}.png".format(f2,'obrazek'), format='png', dpi=100)
        #     print('Image ADDED!')
        # else:
        #     print('Image NOT added!')
        #
        # plt.close('all')
        return [f2]

        # tpch c1 sigma1 c2 sigma2 tpch2
    def get_bounds(self):
        return ([0], [0.5])


def ceff(T, Tpch, Tpch2, c0, c1, c2, sigma1, sigma2):
    return c0 + c1 * np.exp((-(T-Tpch) * (T-Tpch))/sigma1) + c2 * np.exp((-(T-Tpch2) * (T-Tpch2))/sigma2)


algo = pg.algorithm(pg.sade(gen=100))
algo.set_verbosity(1)
prob = pg.problem(heat_f2())
pop = pg.population(prob, 7)
pop = algo.evolve(pop)
# isl = island(algo = de(10), prob = heat_f2(), size=20, udi=thread_island())
# islands = [island(algo = de(gen = 100000, F=effe, CR=cross), prob=heat_f2(), size=20, seed=32) for effe in [0.3,0.5,0.7,0.9] for cross in [0.3,0.5,0.7,0.9]]
# _ = [isl.evolve() for isl in islands]
# _ = [isl.wait() for isl in islands]


d_sol = pop.champion_x[0]
print(d_sol)

# bounds = [(0.005, 0.5, 0.5), (0.025, 1, 1)]

# if __name__ == '__main__':
#     optimizer = ParticleSwarmOptimizer(heat1,
#                                        bounds,
#                                        obj_fct_is_vectorized=False,
#                                        options={'num_particles': 6, 'max_iters': 7})
#     optimizer.optimize()
#
#     argmin = optimizer.historic_best_position
#     minimum = optimizer.historic_best_score
#     print("best arg {} yielded {}".format(argmin, minimum))