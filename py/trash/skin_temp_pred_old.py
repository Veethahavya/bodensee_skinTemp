from scipy import integrate
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import time

#%% Data Import
dfLH = pd.read_csv('..\data\heatflux_latent.dat', sep="\s+", names=['Date','Time', 'LH'])
LH = dfLH['LH'].tolist()

dfSH = pd.read_csv('..\data\heatflux_sfc.dat', sep="\s+", names=['Date','Time', 'SH'])
SH = dfSH['SH'].tolist()

dfLW = pd.read_csv('..\data\longwave_rad_net.dat', sep="\s+", names=['Date','Time', 'LW'])
LW = dfLW['LW'].tolist()

dfSW = pd.read_csv('..\data\shortwave_rad.dat', sep="\s+", names=['Date','Time', 'SW'])
SW = dfSW['SW'].tolist()

dfuwstr = pd.read_csv(r'..\data\ustar.dat', sep="\s+", names=['Date','Time', 'uw_str'])
uw_str = dfuwstr['uw_str'].tolist()

time_check = dfLH['Time'].all() == dfSH['Time'].all() == dfLW['Time'].all()

#%% ODE function definition
#Interpolation function for smaller dt chosen by the solver
def interpolate(lst,t):
    value = 0
    if t == 0:
        value = lst[0]
    if t == len(lst) or t > len(lst):
        value = lst[len(lst)]
    else:
        np.interp(t, list(range(0,len(lst))), lst)
    return value

def func(T, t):
    lh = interpolate(LH, t)
    sh = interpolate(SH, t)
    lw = interpolate(LW, t)
    Q = lh + sh + lw

    a1, a2, a3 = 0.28, 0.27, 0.45
    b1, b2, b3 = 71.5, 2.8, 0.07
    d = 1
    albedo = 0.09
    Rs = (1.0 - albedo) * interpolate(SW, t)
    R_d = (a1*math.exp(-d*b1) + a2*math.exp(-d*b2) + a3*math.exp(-d*b3))*Rs

    ro_w = 1000
    cw = 4174
    v = 0.3

    x = (Q + Rs - R_d)/(d*ro_w*cw*v/(v+1))


    k = 0.4
    alpha_w = 210e-6 #assumed thermal expansion coeff of water at 20C
    g = 9.8
    uw = interpolate(uw_str, t)

    if T > 0:
        Fd = ((v*g*alpha_w/(5*d))**0.5)*ro_w*cw*(uw**2)*(T**0.5)
    else:
        Fd = g * alpha_w * (Q + Rs - R_d)

    L = ro_w*cw*uw**3/(k*Fd)

    if L <= 0:   phi_t_d_L = 1 + 5 * (-d / L)
    else:        phi_t_d_L = (1 - 16 * (-d / L))**-0.5

    c = ((v+1)*k*uw)/(d*phi_t_d_L)


    dTdt = x - c*T
    return dTdt

#%% ODE Solver Init and Run
y0 = [0.001]

t = []
if time_check: t = list(range(len(LH)))

T = integrate.odeint(func, y0, t)

#%%Post-processing
#Save DF output as .dat
dfT = pd.DataFrame(T, columns=['dTdt'])
dfT['Datetime'] = pd.to_datetime(dfLH['Date'] + ' ' + dfLH['Time'], format='%d%m%Y %H:%M:%S', errors='ignore')
dfT = dfT.set_index('Datetime')
dat_path = r'..\results\dTdt_' + time.strftime("%Y%m%d%H%M%S") + '_y0-0.01' + '.dat'
dfT.to_csv(dat_path, sep = ' ')

#Plot and save figure
plt.figure(figsize=(20,6), dpi=120)
#plt.xkcd()
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
dfT.plot(ax=ax)
png_path = r'..\results\dTdt_' + time.strftime("%Y%m%d%H%M%S") + '_y0-0.01' + '.png'
plt.savefig(png_path)


#%% Clearing unwanted variables
del T, ax, dat_path, png_path, t, y0, uw_str, dfLH, dfLW, dfSH, dfSW, dfuwstr