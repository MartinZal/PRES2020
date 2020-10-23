import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pygmo as pg
import string
import re
from heat import fun
from tqdm import tqdm
#from heat.fun import heatIn, qprl, qp, ql, qprp, moveFun1, neighFun, heatInAll, heatPoint, heat, yout
import copy

# <editor-fold desc="PARAMETERS">
l = 0.1
h = 1.5
qh = 0
qd = 0
ro = 760
rhov = 1.1614
lambd = 0.2
T0p = 20
T_set = 35
Nx = 5
Ny = 10
start_hour = 10
end_hour = 13
t_start = 3600 * start_hour
t_end = 3600 * end_hour
tmax = t_end - t_start
qloss = 0
e = 0.03
cp = 1007
mi = 184.6 * 10 ** (-7)
k = 0.0272
cv = 1007
kinv = 15.52 * 10 ** (-6)
m_dot = 0.03
qd = 0
qh = 0
dx = l / (Nx - 1)
dy = h / (Ny - 1)
dd = 0.9
dt = 0.1
v = dy / dt
# </editor-fold>

# <editor-fold desc="FUNCTIONS AND PREPROCESSING">
def Q_conv(n, alpha, dy, dd, Tvz, T, p, Ny):

    if (n == 0):
        Q_conv_out = alpha * dy / 2 * dd * (Tvz[0, p] - T[0, -1, p])
    elif (n == -1):
        Q_conv_out = alpha * dy / 2 * dd * (Tvz[-1, p] - T[0, 0, p])
    else:
        Q_conv_out = alpha * dy * dd * (Tvz[n, p] - T[0, Ny - n, p])

    return Q_conv_out

rad = pd.read_csv('data/17_07_2019_Rad.txt', delimiter=';')



rad_val = rad.values[:, 2].tolist()

def Q_rad(dy, dd, rad_val, p, dt):
    time1 = t_start + p * dt
    n_data = np.floor(time1/60)
    Q_rad_out = (rad_val[int(n_data)] + (rad_val[int(n_data+1)] - rad_val[int(n_data)])*(time1/60 - n_data)) * dy * dd

    return Q_rad_out

Tamb = pd.read_csv('data/17_07_2019_Tamb.txt', delimiter=';')

Tamb_val = Tamb.values[:, 2].tolist()
Tamb_val_clean = []
for i in range(len(Tamb_val)):
    Tamb_val_clean.append(re.sub('\+', '', Tamb_val[i]))
    Tamb_val_clean[i] = Tamb_val_clean[i].replace(',', '.')
    Tamb_val_clean[i] = float(Tamb_val_clean[i])

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.plot(rad.values[:, 2].tolist(), 'r')
# plt.xlabel('Time [min]')
# plt.ylabel('Solar irradiance [W/m^2]')
# plt.subplot(122)
# plt.plot(Tamb_val_clean)
# plt.xlabel('Time [min]')
# plt.ylabel('Ambient temperature [°C]')
# plt.show()
# plt.savefig("Images/input.png", format='png', dpi=300)

def T_ambient(Tamb_val_clean, p, dt):
    time1 = t_start + p * dt
    n_data = np.floor(time1/60)
    T_amb_out = Tamb_val_clean[int(n_data)] + (Tamb_val_clean[int(n_data+1)] - Tamb_val_clean[int(n_data)])*(time1/60 - n_data)

    return T_amb_out


Q_loss = 0
# def ql(p, T, Nzv, Tvz, alpha):
#     return alpha * (Tvz[0:Nzv, p, 0] - T[0, :, p]) #+ 1000*0.91*0.95*1000
#
# def qprl(n, p, T, Tvz, alpha, neighField):
#     return alpha * np.dot(neighField[:, n], (T[0, :, p] - Tvz[0:Nzv, p, 0]))



def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis', interpolation='spline16')
    plt.colorbar()
    plt.show()
# </editor-fold>

#MAIN FUNCTION
def heat_2d(x):
    Tvz_out = []
    alpha = fun.calcDN(h, e, cv, mi, k, v, kinv)
    fitness = 0
    c0 = 2000
    c1 = x[2] #61300
    Tpch = x[1]
    sigma = 0.5

    def c(T):
        return c0 + c1 * np.exp((-(T - Tpch) * (T - Tpch)) / sigma)

    dx = x[0] / (Nx - 1)
    dy = h / (Ny - 1)

    Tvz = np.zeros([Ny, int(tmax / dt) + 1])
    T = np.ones([Nx, Ny, int(tmax / dt) + 1])
    T[:, :, 0].fill(T0p)
    Tvz[:].fill(T_ambient(Tamb_val_clean, 0, dt))
    qp = 0
    Q_sum = np.zeros(int(tmax / dt))
    for p in tqdm(np.arange(0, int(tmax / dt), 1)):

        #vzduch
        for i in range(Ny):
            Tvz[i, p + 1] = Tvz[i, p] - Q_conv(i, alpha, dy, dd, Tvz, T, p, Ny) / (m_dot * cv) - Q_loss / (m_dot * cv)
            Q_sum[p] = Q_sum[p] + Q_conv(i, alpha, dy, dd, Tvz, T, p, Ny)

        #krajni body

        #levy horni
        T[0, 0, p + 1] = T[0, 0, p] + (2 * dt * qh) / (ro * dy * c(T[0, 0, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[0, 0, p])) * (T[0, 1, p] - T[0, 0, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, 0, p])) * (T[1, 0, p] - T[0, 0, p])  \
                          + (2 * dt) / (ro * dy * dx * c(T[0, 0, p])) * (Q_conv(0, alpha, dy, dd, Tvz, T, p, Ny) + Q_rad(dy, dd, rad_val, p, dt) / 2)

        #levy dolni
        T[0, -1, p + 1] = T[0, -1, p] + (2 * dt * qd) / (ro * dy * c(T[0, -1, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[0, -1, p])) * (T[0, -1 - 1, p] - T[0, -1, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, -1, p])) * (T[1, -1, p] - T[0, -1, p])  \
                          + (2 * dt) / (ro * dy * dx * c(T[0, -1, p])) * (Q_conv(-1, alpha, dy, dd, Tvz, T, p, Ny) + Q_rad(dy, dd, rad_val, p, dt) / 2)

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


                #cyklus pro levou hranu - under correction
                T[0, n, p + 1] = T[0, n, p] + (lambd * dt) / (ro * dy ** 2 * c(T[0, n, p])) * (T[0, n + 1, p] + T[0, n - 1, p] - 2 * T[0, n, p]) + \
                                 (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, n, p])) * (T[1, n, p] - T[0, n, p]) + \
                                 (2 * dt) / (ro * dx * dy * c(T[0, n, p])) * (Q_conv(n, alpha, dy, dd, Tvz, T, p, Ny) + Q_rad(dy, dd, rad_val, p, dt))

                #cyklus pro pravou hranu
                T[-1, n, p + 1] = T[-1, n, p] + (2 * dt * qp) / (ro * dy * c(T[-1, n, p])) + (lambd * dt) / (ro * dy ** 2 * c(T[-1, n, p])) * (T[-1, n + 1, p] + T[-1, n - 1, p] - 2 * T[-1, n, p]) + \
                                 (2 * lambd * dt) / (ro * dx ** 2 * c(T[-1, n, p])) * (T[-1 -1, n, p] - T[-1, n, p])
                #stred
                T[m, n, p + 1] = T[m, n, p] + (lambd * dt) / (ro * dy ** 2 * c(T[m, n, p])) * (T[m, n + 1, p] + T[m, n - 1, p] - 2*T[m, n, p]) +\
                                 (lambd * dt) / (ro * dx ** 2 * c(T[m, n, p])) * (T[m + 1, n, p] + T[m - 1, n, p] - 2*T[m, n, p])

        Tvz[:, p + 1] = np.roll(Tvz[:, p + 1], 1)
        Tvz_out.append(Tvz[0, p + 1])
        Tvz[0, p + 1] = T_ambient(Tamb_val_clean, p, dt)
        if (p>Ny):
            fitness = fitness + (Tvz[-1, p + 1] - T_set)**2

    return T, Tvz, Tvz_out, fitness, Q_sum #

#EFFECTIVE HEAT CAPACITY
def ceff(T, c0, c1, Tpch, sigma):
    return c0 + c1 * np.exp((-(T - Tpch) * (T - Tpch)) / sigma)

#2D PLOT
# plt.figure()
# plt.plot(Tvz_out[0:37000:600])
# plt.plot(T_out[1, 1, 0:37000:600])
# plt.plot(Tamb_val_clean[60*12:60*13])
# plt.plot(rad_val[60*12:60*13])
# plt.legend(['Outlet - Air temperature', 'PCM surface temperature', 'Inlet - Ambient temperature', 'Solar irradiation'])
# plt.show()

# fig, ax = plt.subplots()
#
# for i in range(len(T_out[0, 0, :])):
#     ax.cla()
#     ax.set_title("frame {}".format(i))
#     heatmap2d(np.transpose(np.vstack((Tvz[:, i], T_out[:, :, i]))))
#
#     plt.pause(0.1)
#     plt.clf()
# #plt.imshow(T[:, :, 600])

# <editor-fold desc="FITNESS CLASS & PLOTS">
counter = 0
list = []
class heat_f2:
    counter = 0
    def fitness(self, x):
        global counter, list
        counter = counter + 1
        start = time.time()
        T_out, Tvz, Tvz_out, f2, Q_sum = heat_2d(x)
        end = time.time()
        c0 = 2000
        c1 = x[2]
        sigma = 0.5


        # print('Iteration number: ' + str(counter))
        # print('Elapsed time: ' + str(end - start))
        # print('Fitness: ' + str(f2))
        # print('Variables: Tpch, Tpch2, c0, c1, c2, sigma1, sigma2')
        # print('Current x: ' + str(x))
        # print('----------------------------------------------------------------')
        # list.append(f2)

        # plt.figure()
        # plt.plot(Tvz_out[0:-1])
        # plt.plot(T_out[1, 1, 0:-1])
        # plt.plot(Tamb_val_clean[start_hour])
        # #plt.plot(rad_val[60 * start_hour:int(60 * end_hour)])
        # plt.legend(['Outlet - Air temperature', 'PCM surface temperature', 'Inlet - Ambient temperature', 'Solar irradiation'])
        # plt.show()
        plt.figure(figsize=(20, 12))
        plt.subplot(231)
        #plt.figure(figsize=(17, 10), dpi=150)#, facecolor='w', edgecolor='k'

        plt.plot(Tvz_out[0:-1:600])
        plt.plot(T_out[0, -1, 0:-1:600])
        plt.plot(Tamb_val_clean[60 * start_hour: int(60 * end_hour)])
        #plt.plot(rad_val[60 * start_hour:int(60 * end_hour)])

        plt.xlabel('Time [min]')
        plt.ylabel('Temperature [°C]')
        plt.title('Thickness: {:0.3f} m, Tpch = {:02.02f} °C, c1 = {:05.02f}'.format(x[0], x[1], x[2]))

        plt.legend(['Outlet - Air temperature', 'PCM surface temperature', 'Inlet - Ambient temperature'])



        # plt.figure(num=None, figsize=(17, 7), dpi=100, facecolor='w', edgecolor='k')
        # plt.subplot(1, 2, 1)
        # plt.plot(np.arange(t_start, t_max, dt), T[5, :p_max], np.arange(t_start, t_max, dt), core[:p_max])
        # plt.xlabel('Time iteration []')
        # plt.ylabel('Temperature [°C]')
        # plt.title('Temp. evolution comparison, middle of PCM layer-experiment vs simulation')
        # plt.legend(['Simulation', 'Experiment'])
        # print(T[5, :])

        plt.subplot(232)
        plt.plot(Tvz_out[0:-1:600])
        plt.plot(np.linspace(0, tmax/60, np.size(Tvz_out)), np.ones(np.size(Tvz_out)) * T_set,'--r')
        plt.xlabel('Time [min]')
        plt.ylabel('Temperature [°C]')
        plt.legend(['T_out - Outlet air temperature', 'T_set - Set outlet air temperature'])

        plt.subplot(233)
        plt.plot(np.linspace(x[1]-20, x[1]+20, 200), ceff(np.linspace(x[1]-20, x[1]+20, 200), c0, c1, x[1], sigma))
        plt.xlabel('Temperature [°C]')
        plt.ylabel('Effective heat capacity [J/(kg K)]')
        plt.title('Opt ceff = {:04.0f} + {:05.0f}exp(-(T - {:02.2f})^2)/{:02.2f}'.format(c0, c1, x[1], sigma))

        plt.subplot(234)
        plt.plot(rad_val[60 * start_hour:int(60 * end_hour)], 'r') #rad_val[60 * start_hour:int(60 * end_hour)]
        plt.xlabel('Time [min]')
        plt.ylabel('Solar irradiance [W/m^2]')

        plt.subplot(235)
        plt.plot(np.linspace(0, tmax/60, np.size(Q_sum)), Q_sum) #np.linspace(0, tmax/60, np.size(Q_sum)
        plt.xlabel('Time [min]')
        plt.ylabel('Convective heat flux into PCM [J/min]')
        #print(Q_sum[-10:])

        # res = (Tvz_out[0:-1:600] - np.ones(np.size(Tvz_out))

        plt.subplot(236)
        plt.plot(np.linspace(0, tmax / 60, np.size(Tvz_out[0:-1:600])), (Tvz_out[0:-1:600] - np.ones(np.size(Tvz_out[0:-1:600])) * T_set), '--g')
        plt.xlabel('Time [min]')
        plt.ylabel('Temperature residuum [°C]')

        # plt.figure()
        # plt.plot(np.linspace(41 - 20, 41 + 20, 200),
        #          ceff(np.linspace(41 - 10, 41 + 10, 200), 2000, 61300, 41, 2.1))
        # plt.xlabel('Temperature [°C]')
        # plt.ylabel('Effective heat capacity [J/(kg K)]')
        # plt.title('ceff(T) = {:04.0f} + {:05.0f}exp(-(T - {:02.2f})^2)/{:02.2f}'.format(2000, 61300, 41, 2.1))

        plt.show()
        print(list)


        if (True):
            plt.savefig("Images/{:02.0f}{}.png".format(f2, 'fitness'), format='png', dpi=300)
            print('Image ADDED!')
        else:
            print('Image NOT added!')

        plt.close('all')
        return [f2]

        # tpch c1 sigma1 c2 sigma2 tpch2
    def get_bounds(self):
        return ([0.001, 35, 30000], [0.1, 75, 70000])
# </editor-fold>

# <editor-fold desc="OPTIMIZATION">
algo = pg.algorithm(pg.sade(gen=100, ftol=100))
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
# </editor-fold>
