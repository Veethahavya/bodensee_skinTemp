from scipy import integrate
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

dfLH = pd.read_csv('..\data\heatflux_latent.dat', sep="\s+", names=['Date','Time', 'LH'])
LH = dfLH['LH'].tolist()

dfSH = pd.read_csv('..\data\heatflux_sfc.dat', sep="\s+", names=['Date','Time', 'SH'])
SH = dfSH['SH'].tolist()

dfLW = pd.read_csv('..\data\longwave_rad.dat', sep="\s+", names=['Date','Time', 'LW'])
LW = dfLW['LW'].tolist()

dfSW = pd.read_csv('..\data\shortwave_rad.dat', sep="\s+", names=['Date','Time', 'SW'])
SW = dfSW['SW'].tolist()

dfuwstr = pd.read_csv(r'..\data\ustar.dat', sep="\s+", names=['Date','Time', 'uw_str'])
uw_str = dfuwstr['uw_str'].tolist()

time_check = dfLH['Time'].all() == dfSH['Time'].all() == dfLW['Time'].all()

def func(T,t):
    Q = LH[int(t)] + SH[int(t)] + LW[int(t)]

    a1, a2, a3 = 0.28, 0.27, 0.45
    b1, b2, b3 = 71.5, 2.8, 0.07
    d = 2
    R_d = a1*math.exp(-d*b1) + a2*math.exp(-d*b2) + a3*math.exp(-d*b3)
    Rs = LW[int(t)] + SW[int(t)]

    ro_w = 1000
    cw = 4174
    v = 0.3

    x = (Q + Rs - R_d)/(d*ro_w*cw*v/(v+1))

    k = 0.4
    alpha_w = 210e-6 #assumed thermal expansion coeff of water at 20C
    g = 9.8
    Fd = ((v*g*alpha_w/5*d)**0.5)*ro_w*cw*uw_str[int(t)]**2*T
    L = ro_w*cw*uw_str[int(t)]**3/(k*Fd)
    phi_t_d_L = 5*d/L
    c = ((v+1)*k*uw_str[int(t)])/(d*phi_t_d_L)

    dTdt = x - c*T

    return dTdt

y0 = 1e-1
t = []
if time_check: t = list(range(len(LH)))
T = integrate.odeint(func, y0, t)

dfT = pd.DataFrame(T, columns=['dTdt'])
dfT['Datetime'] = pd.to_datetime(dfLH['Date'] + ' ' + dfLH['Time'], format='%d%m%Y %H:%M:%S', errors='ignore')
dfT = dfT.set_index('Datetime')

time = np.array(dfLH['Time'].tolist())
plt.figure(figsize=(20,6))
#plt.xkcd()
plt.plot(dfT)

dfT.to_csv("dTdt.dat", sep = " ")