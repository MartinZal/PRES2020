import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import pygmo as pg
import string
import re
from heat import fun
#from heat.fun import heatIn, qprl, qp, ql, qprp, moveFun1, neighFun, heatInAll, heatPoint, heat, yout
import copy

# <editor-fold desc="PARAMETERS">
l = 0.01
h = 1.5
qh = 0
qd = 0
ro = 880
rhov = 1.16
lambd = 0.2
T0p = 20
T_set = 40
Nx = 5
Ny = 5
start_hour = 10
end_hour = 14
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
m_dot = 0.038
qd = 0
qh = 0
dx = l / (Nx - 1)
dy = h / (Ny - 1)
dd = 0.9
dt = 0.1
v = 4.18

#dy/dt
# </editor-fold>

# <editor-fold desc="FUNCTIONS AND PREPROCESSING">
def Q_conv(n, alpha, dy, Tvz, T, p):

    Q_conv_out = alpha * dy * (Tvz[n, p] - T[0, n, p])

    return Q_conv_out

def Q_conv1(n, alpha, dy, Tvz1, T, p):

    Q_conv_out = alpha * dy * (Tvz1[n, p] - T[-1, n, p])

    return Q_conv_out


validation = pd.read_csv('data/validation.csv', delimiter=';')
T_air_in_valid = validation.values[:, 1].tolist()
T_air_out_valid = validation.values[:, 2].tolist()
rad_valid = validation.values[:, 3].tolist()


T_air_in_V, T_air_out_V, rad_V = [], [], []
for i in range(np.size(T_air_in_valid)):
    if ((i % 4) == 0):
        T_air_in_V.append(T_air_in_valid[i])
        T_air_out_V.append(T_air_out_valid[i])
        rad_V.append(rad_valid[i])
T_air_in_V = T_air_in_V[:610]
T_air_out_V = T_air_out_V[:610]
rad_V = rad_V[:610]


rad = pd.read_csv('data/17_07_2019_Rad.txt', delimiter=';')

rad_val = rad.values[:, 2].tolist()

def Q_rad(dy, dd, rad_val, p, dt):
    time1 = t_start + p * dt
    n_data = np.floor(time1/60)
    Q_rad_out = (rad_val[int(n_data)] + (rad_val[int(n_data+1)] - rad_val[int(n_data)])*(time1/60 - n_data)) * dy

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



def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis', interpolation='spline16')
    plt.colorbar()
    plt.show()
# </editor-fold>

#MAIN FUNCTION
def heat_2d(x):
    Tvz_out = []
    Tvz_out1 = []
    Tvz_out_final = []
    #alpha = fun.calcDN(h, e, cv, mi, k, v, kinv)
    alpha = 40
    print('Alpha is:' + str(alpha))
    fitness = 0
    c0 = 2000
    c1 = x[2] #61300
    Tpch = x[1]
    sigma = 2.1
    qp = 0

    def c(T):
        return c0 + c1 * np.exp((-(T - Tpch) * (T - Tpch)) / sigma)

    dx = x[0] / (Nx - 1)
    dy = h / (Ny - 1)

    Tvz = np.zeros([Ny, int(tmax / dt) + 1])
    Tvz1 = np.zeros([Ny, int(tmax / dt) + 1])

    T = np.ones([Nx, Ny, int(tmax / dt) + 1])
    T[:, :, 0].fill(T_ambient(Tamb_val_clean, 0, dt))

    Tvz[:].fill(T_ambient(Tamb_val_clean, 0, dt))
    Tvz1[:].fill(T_ambient(Tamb_val_clean, 0, dt))


    Q_sum = np.zeros(int(tmax / dt))
    Q_sum1 = np.zeros(int(tmax / dt))

    for p in tqdm(np.arange(0, int(tmax / dt), 1)):

        #vzduch
        for i in range(Ny):
            Tvz[i, p + 1] = Tvz[i, p] - Q_conv(i, alpha, dy, Tvz, T, p) / (m_dot * cv) # - Q_loss / (m_dot * cv)
            Q_sum[p] = Q_sum[p] + Q_conv(i, alpha, dy, Tvz, T, p)

            Tvz1[i, p + 1] = Tvz1[i, p] - Q_conv1(i, alpha, dy, Tvz1, T, p) / (m_dot * cv) #- Q_loss / (m_dot * cv)
            Q_sum1[p] = Q_sum1[p] + Q_conv1(i, alpha, dy, Tvz1, T, p,)

        #krajni body

        #levy horni
        T[0, 0, p + 1] = T[0, 0, p] + (2 * dt * qh) / (ro * dy * c(T[0, 0, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[0, 0, p])) * (T[0, 1, p] - T[0, 0, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, 0, p])) * (T[1, 0, p] - T[0, 0, p])  \
                          + (2 * dt) / (ro * dy * dx * c(T[0, 0, p])) * (Q_conv(0, alpha, dy, Tvz, T, p) + Q_rad(dy, dd, rad_val, p, dt))

        #levy dolni
        T[0, -1, p + 1] = T[0, -1, p] + (2 * dt * qd) / (ro * dy * c(T[0, -1, p])) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[0, -1, p])) * (T[0, -1 - 1, p] - T[0, -1, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[0, -1, p])) * (T[1, -1, p] - T[0, -1, p])  \
                          + (2 * dt) / (ro * dy * dx * c(T[0, -1, p])) * (Q_conv(-1, alpha, dy, Tvz, T, p) + Q_rad(dy, dd, rad_val, p, dt))

        #pravy horni
        T[-1, 0, p + 1] = T[-1, 0, p] + (2 * dt * qh) / (ro * dy * c(T[-1, 0, p])) + (2 * dt) / (ro * dx * dy * c(T[-1, 0, p]))* (Q_conv1(0, alpha, dy, Tvz1, T, p)) + \
                         (2 * lambd * dt) / (ro * dy ** 2 * c(T[-1, 0, p])) * (T[-1, 1, p] - T[-1, 0, p]) + (2 * lambd * dt) / (ro * dx ** 2 * c(T[-1, 0, p])) * (T[-1-1, 0, p] - T[-1, 0, p])

        #pravy dolni
        T[-1, -1, p + 1] = T[-1, -1, p] + (2 * dt * qd) / (ro * dy * c(T[-1, -1, p])) + (2 * dt) / (ro * dx * dy * c(T[-1, -1, p])) * (Q_conv1(-1, alpha, dy, Tvz1, T, p)) + \
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
                                 (2 * dt) / (ro * dy * dx * c(T[0, n, p])) * (Q_conv(n, alpha, dy, Tvz, T, p) + Q_rad(dy, dd, rad_val, p, dt))

                #cyklus pro pravou hranu
                T[-1, n, p + 1] = T[-1, n, p] + (2 * dt) / (ro * dy * dx * c(T[-1, n, p])) * (Q_conv1(n, alpha, dy, Tvz1, T, p)) + \
                                  (lambd * dt) / (ro * dy ** 2 * c(T[-1, n, p])) * (T[-1, n + 1, p] + T[-1, n - 1, p] - 2 * T[-1, n, p]) + \
                                 (2 * lambd * dt) / (ro * dx ** 2 * c(T[-1, n, p])) * (T[-1 -1, n, p] - T[-1, n, p])

                #stred
                T[m, n, p + 1] = T[m, n, p] + (lambd * dt) / (ro * dy ** 2 * c(T[m, n, p])) * (T[m, n + 1, p] + T[m, n - 1, p] - 2*T[m, n, p]) +\
                                 (lambd * dt) / (ro * dx ** 2 * c(T[m, n, p])) * (T[m + 1, n, p] + T[m - 1, n, p] - 2*T[m, n, p])

        Tvz_out_final.append((Tvz[-1, p + 1] + Tvz1[-1, p + 1]) / 2)

        Tvz[:, p + 1] = np.roll(Tvz[:, p + 1], 1)
        Tvz_out.append(Tvz[0, p + 1])
        Tvz[0, p + 1] = T_ambient(Tamb_val_clean, p, dt)

        Tvz1[:, p + 1] = np.roll(Tvz1[:, p + 1], 1)
        Tvz_out1.append(Tvz1[0, p + 1])
        Tvz1[0, p + 1] = T_ambient(Tamb_val_clean, p, dt)

        if (p>600*5):
            fitness = fitness + (Tvz_out_final[p] - T_set)**2

    return T, Tvz, Tvz_out_final, Tvz_out, Tvz_out1, fitness, Q_sum, Q_sum1 #

#EFFECTIVE HEAT CAPACITY
def ceff(T, c0, c1, Tpch, sigma):
    return c0 + c1 * np.exp((-(T - Tpch) * (T - Tpch)) / sigma)

# <editor-fold desc="FITNESS CLASS & PLOTS">
counter = 0
list = []
class heat_f2:
    counter = 0
    def fitness(self, x):
        global counter, list
        counter = counter + 1
        start = time.time()
        T_out, Tvz, Tvz_out, Tvz_left, Tvz_right, f2, Q_sum, Q_sum1 = heat_2d(x)
        end = time.time()
        c0 = 2000
        c1 = x[2]
        sigma = 0.5


        print('Iteration number: ' + str(counter))
        print('Elapsed time: ' + str(end - start))
        print('Fitness: ' + str(f2))
        print('Variables: Tpch, Tpch2, c0, c1, c2, sigma1, sigma2')
        print('Current x: ' + str(x))
        print('----------------------------------------------------------------')
        list.append(f2)

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
            plt.savefig("ImagesFinal/{:02.0f}{}.png".format(f2, 'fitness'), format='png', dpi=300)
            print('Image ADDED!')
        else:
            print('Image NOT added!')

        #plt.close('all')
        return [f2]

        # tpch c1 sigma1 c2 sigma2 tpch2
    def get_bounds(self):
        return ([0.001, 35, 30000], [0.1, 75, 70000])
# </editor-fold>

# <editor-fold desc="OPTIMIZATION">
algo = pg.algorithm(pg.sade(gen=100, ftol=100))
algo.set_verbosity(1)
prob = pg.problem(heat_f2())
pop = pg.population(prob, 10)
pop = algo.evolve(pop)
# isl = island(algo = de(10), prob = heat_f2(), size=20, udi=thread_island())
# islands = [island(algo = de(gen = 100000, F=effe, CR=cross), prob=heat_f2(), size=20, seed=32) for effe in [0.3,0.5,0.7,0.9] for cross in [0.3,0.5,0.7,0.9]]
# _ = [isl.evolve() for isl in islands]
# _ = [isl.wait() for isl in islands]


d_sol = pop.champion_x[0]
print(d_sol)
# </editor-fold>






# T_out, Tvz, Tvz_out, Tvz_left, Tvz_right, f2, Q_sum, Q_sum1 = heat_2d(x)
#
# plt.figure()
# plt.figure(figsize=(20, 12))
#
#
# plt.subplot(221)
# fig, ax1 = plt.subplots()
# color = 'tab:blue'
# ax1.set_xlabel('time (s)')
#
# ax1.tick_params(axis='y', labelcolor=color)
#
#
# ax1.plot(Tvz_out[0:-1:600])
#
# ax1.plot(T_air_in_V[60 * start_hour: int(60 * end_hour)])
# ax1.plot(T_air_out_V[60 * start_hour: int(60 * end_hour)])
#
# ax2 = ax1.twinx()
#
# color = 'tab:red'
# ax2.set_ylabel('radiation', color=color)  # we already handled the x-label with ax1
# ax2.plot(rad_V[60 * start_hour: int(60 * end_hour)], 'r--')
#
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()
#
# plt.show()
# # plt.plot(Tvz_left[0:-1:600])
# # plt.plot(Tvz_right[0:-1:600])
#
# #plt.plot(rad_val[60 * start_hour:int(60 * end_hour)])
#
# plt.xlabel('Time [min]')
# plt.ylabel('Temperature [°C]')
# plt.title('Thickness: {:0.3f} m, Tpch = {:02.02f} °C, c1 = {:05.02f}'.format(x[0], x[1], x[2]))
#
# plt.legend(['Outlet air temperature - simulation', 'Ambient temperature - experiment', 'Outlet air temperature - experiment'])
#
# plt.subplot(222)
#
# plt.plot(Q_sum)
# plt.plot(Q_sum1)
#
# plt.subplot(223)
# plt.plot(T_air_in_V[60 * start_hour: int(60 * end_hour)])
# plt.plot(T_air_out_V[60 * start_hour: int(60 * end_hour)])
# plt.plot(rad_V[60 * start_hour: int(60 * end_hour)])
#
#
#
# plt.show()


