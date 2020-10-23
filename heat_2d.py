import numpy as np
import matplotlib.pyplot as plt
from heat import fun
#from heat.fun import heatIn, qprl, qp, ql, qprp, moveFun1, neighFun, heatInAll, heatPoint, heat, yout
import copy

l = 0.1
h = 1
qh = 0
qd = 0
ro = 760
rhov = 1.1614
lambd = 0.2
T0p = 1500
T0v = 18
Nx = 10
Ny = 21
Nzv = 21
dt = 0.1
tmax = 60
qloss = 0
e = 0.03
cp = 1007
mi = 184.6 * 10 ** (-7)
k = 0.0272
cv = 1007
kinv = 15.52 * 10 ** (-6)

dy = h / (Ny - 1)
v = 0.8

# predpocet intevalu
a = np.zeros([2, Ny])
a[0, 0] = 0
a[1, -1] = h
for i in range(0, Ny - 1):
    a[1, i] = i * dy + dy / 2
    a[0, i + 1] = a[1, i]

dzv = h / (Nzv - 1)

b = np.zeros([2, Nzv])
b[0, 0] = 0
b[1, -1] = h

for i in range(0, Nzv - 1):
    b[1, i] = i * dzv + dzv / 2
    b[0, i + 1] = b[1, i]

neighField = fun.neighFun(a, b, Ny, Nzv)


#ql = 1000000
#qp = 5000000
qd = 0
qh = 0


dx = l / (Nx - 1)
dy = h / (Ny - 1)
dd = 0.5

def ql(n, p, T, Nzv, Tvz, alpha, neighField):
    return alpha * np.dot(neighField[n, :], (Tvz[0:Nzv, p, 0] - T[0, :, p])) #+ 1000*0.91*0.95*1000

def qprl(n, p, T, Tvz, alpha, neighField):
    return alpha * np.dot(neighField[:, n], (T[0, :, p] - Tvz[0:Nzv, p, 0]))

def c(T):
    return 2000 + 40000 * np.exp((-(T-20) * (T-20))/3)

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis', interpolation='spline16')
    plt.colorbar()
    plt.show()

def heat_2d():
    dyp = dt * v
    dy = h / (Ny - 1)
    if (dyp >= dy):
        print('Neni splnena podminka na rychlost, muze dojit k preskoceni jednoho KO')

    alpha = 100#fun.calcDN(h, e, cv, mi, k, v, kinv)

    dx = l / (Nx - 1)
    dy = h / (Ny - 1)

    Tvz = np.zeros([15000, int(tmax / dt) + 2, 2])
    T = np.ones([Nx, Ny, int(tmax / dt) + 1])
    T[:, :, 0].fill(T0p)
    Tvz[:, :, :].fill(T0v)
    qp = 0
    bIt = copy.copy(b)
    neighField = fun.neighFun(a, b, Ny, Nzv)

    for p in np.arange(0, int(tmax / dt), 1):

        #??????????
        Tvz[0, p + 1, 0] = Tvz[0, p, 0] + (2 * dt) / (e * dd * neighField[0, 0] * cv * rhov) * \
        (qprl(0, p, T, Tvz, alpha, neighField) + qloss)

        for i in range(1, Nzv - 1):
            Tvz[i, p + 1, 0] = Tvz[i, p, 0] + dt / (ro * cv * dy * e * dd) * (dd * qprl(i, p, T, Tvz, alpha, neighField) + dd * dy * qloss)

        Tvz[Nzv, p + 1, 0] = Tvz[-1, p, 0] + (2 * dt) / (e * dd * neighField[0, 0] * cv * rhov) * \
        (qprl(-1, p, T, Tvz, alpha, neighField) + qloss)

        #krajni body

        #levy horni
        T[0, 0, p + 1] = T[0, 0, p] + dd * (2 * dt * ql(0, p, T, Nzv, Tvz, alpha, neighField)) / (ro * dy * dx * c(T[0, 0, p])) + (2 * dt * qh) / (ro * dy * c(T[0, 0, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[0, 0, p])) * (T[0, 1, p] - T[0, 0, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, 0, p])) * (T[1, 0, p] - T[0, 0, p])

        #levy dolni
        T[0, -1, p + 1] = T[0, -1, p] + dd * (2 * dt * ql(0, p, T, Nzv, Tvz, alpha, neighField)) / (ro * dy * dx * c(T[0, -1, p])) + (2 * dt * qd) / (ro * dy * c(T[0, -1, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[0, -1, p])) * (T[0, -1 - 1, p] - T[0, -1, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, -1, p])) * (T[1, -1, p] - T[0, -1, p])

        #pravy horni
        T[-1, 0, p + 1] = T[-1, 0, p] + (2 * dt * qh) / (ro * dy * c(T[-1, 0, p])) + (2 * dt * qp) / (ro * dx * c(T[-1, 0, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[-1, 0, p])) * (T[-1, 1, p] - T[-1, 0, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[-1, 0, p])) * (T[-1-1, 0, p] - T[-1, 0, p])

        #pravy dolni
        T[-1, -1, p + 1] = T[-1, -1, p] + (2 * dt * qd) / (ro * dy * c(T[-1, -1, p])) + (2 * dt * qp) / (ro * dx * c(T[-1, -1, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[-1, -1, p])) * (T[-1, -1-1, p] - T[-1, -1, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[-1, -1, p])) * (T[-1-1, -1, p] - T[-1, -1, p])

        for m in range(1, Nx - 1):

            #cyklus pro dolni hranu
            T[m, -1, p + 1] = T[m, -1, p] + (2 * dt * qd) / (ro * dy * c(T[m, -1, p])) + (lambd * dt) / (ro * dx ** 2 * c(T[m, -1, p])) * (T[m + 1, -1, p] + T[m - 1, -1, p] - 2*T[m, -1, p]) +\
                             (2 * lambd * dt) / (ro * dy ** 2 * c(T[m, -1, p])) * (T[m, 1, p] - T[m, 0, p])

            #cyklus pro horni hranu
            T[m, 0, p + 1] = T[m, 0, p] + (2 * dt * qh) / (ro * dy * c(T[m, 0, p])) + (lambd * dt) / (ro * dx ** 2 * c(T[m, 0, p])) * (T[m + 1, 0, p] + T[m - 1, 0, p] - 2*T[m, 0, p]) +\
                             (2 * lambd * dt) / (ro * dy ** 2 * c(T[m, 0, p])) * (T[m, -1 - 1, p] - T[m, -1, p])

            for n in range(1, Ny - 1):


                #cyklus pro levou hranu
                T[0, n, p + 1] = T[0, n, p] + dd * (2 * dt * ql(0, p, T, Nzv, Tvz, alpha, neighField)) / (ro * dx * dy * c(T[0, n, p])) + (lambd * dt) / (ro * dy ** 2 * c(T[0, n, p])) * (T[0, n + 1, p] + T[0, n - 1, p] - 2 * T[0, n, p]) + \
                                 (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, n, p])) * (T[1, n, p] - T[0, n, p])
                #cyklus pro pravou hranu
                T[-1, n, p + 1] = T[-1, n, p] + (2 * dt * qp) / (ro * dy * c(T[-1, n, p])) + (lambd * dt) / (ro * dy ** 2 * c(T[-1, n, p])) * (T[-1, n + 1, p] + T[-1, n - 1, p] - 2 * T[-1, n, p]) + \
                                 (2 * lambd * dt) / (ro * dx ** 2 * c(T[-1, n, p])) * (T[-1 -1, n, p] - T[-1, n, p])
                #stred
                T[m, n, p + 1] = T[m, n, p] + (lambd * dt) / (ro * dy ** 2 * c(T[m, n, p])) * (T[m, n + 1, p] + T[m, n - 1, p] - 2*T[m, n, p]) +\
                                 (lambd * dt) / (ro * dx ** 2 * c(T[m, n, p])) * (T[m + 1, n, p] + T[m - 1, n, p] - 2*T[m, n, p])
        bN, nout = fun.moveFun1(bIt, dzv, dyp, Nzv, h)
        bIt = bN
        Tvz[:, p + 1] = np.roll(Tvz[:, p + 1], nout, axis=1)
        Tvz[:, p + 2] = np.roll(Tvz[:, p + 1], nout, axis=1)
    return T, Tvz


T_out, Tvz = heat_2d()



fig, ax = plt.subplots()

for i in range(len(T_out[0, 0, :])):
    ax.cla()
    ax.set_title("frame {}".format(i))
    heatmap2d(np.transpose(np.vstack((Tvz[:Nzv, i, 0], T_out[:, :, i]))))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.1)
    plt.clf()
#plt.imshow(T[:, :, 600])
