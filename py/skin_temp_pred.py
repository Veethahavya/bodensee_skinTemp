import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import math, time, sys, datetime

#%% Data Reads
inp={}

dfLH = pd.read_csv('..\data\heatflux_latent.dat', sep="\s+", names=['Date','Time', 'LH'])
inp['LH'] = dfLH['LH'].tolist()
dfLH['Datetime'] = pd.to_datetime(dfLH['Date'] + ' ' + dfLH['Time'], format='%d%m%Y %H:%M:%S', errors='ignore')
dfLH = dfLH.set_index('Datetime')

dfSH = pd.read_csv('..\data\heatflux_sfc.dat', sep="\s+", names=['Date','Time', 'SH'])
inp['SH'] = dfSH['SH'].tolist()

dfLW = pd.read_csv('..\data\longwave_rad_net.dat', sep="\s+", names=['Date','Time', 'LW'])
inp['LW'] = dfLW['LW'].tolist()

dfSW = pd.read_csv('..\data\shortwave_rad.dat', sep="\s+", names=['Date','Time', 'SW'])
inp['SW'] = dfSW['SW'].tolist()

dfU_str = pd.read_csv(r'..\data\ustar.dat', sep="\s+", names=['Date','Time', 'U_str'])
inp['U_str'] = dfU_str['U_str'].tolist()

dfTb = pd.read_csv(r'..\data\wt_1m.dat', sep="\s+", names=['Date','Time', 'Tb'])
inp['Tb'] = dfTb['Tb'].tolist()

time_check = dfLH['Time'].all() == dfSH['Time'].all() == dfLW['Time'].all()
if not time_check:
    print('Time series of the input data do not match. Exiting.')
    sys.exit(0)
del time_check

#%% Time-step discretization
n_step = 10000
start_time = time.mktime(datetime.datetime.strptime(dfLH.index[0], '%d.%m.%Y %H:%M:%S').timetuple())
end_time = time.mktime(datetime.datetime.strptime(dfLH.index[-1], '%d.%m.%Y %H:%M:%S').timetuple())
t = np.linspace(start_time, end_time, num=n_step, endpoint=True)

#%% Interpolation function for smaller dt than the one in the input data time series
def interpolate_inp(lst,t):
    value = np.interp(t, np.linspace(start_time,end_time, num=len(lst), endpoint=True), lst)
    return value

def interpolate_res(lst, x):
    value = np.interp(x, list(range(0, len(lst))), lst)
    return value

#%% Skin Temperature Calculation
const = {}
const['d']  = 1
const['znuw'] = 1.e-6
const['zkw'] = 1.4e-7
const['g'] = 9.8
const['rhow'] = 1025
const['rho'] = 1.2
const['cw'] = 4190
const['an'] = 0.3
const['zk'] = 0.4
res = {}
res['Tw']=[]
res['Tc']=[]
res['Ts']=[]
for i in range(0,len(t)):
    delt = t[i] - t[i-1]
    var = {}
    inp['lh'] = interpolate_inp(inp['LH'], t[i])
    inp['sh'] = interpolate_inp(inp['SH'], t[i])
    inp['lw'] = interpolate_inp(inp['LW'], t[i])
    var['qo'] = inp['lh'] + inp['sh'] + inp['lw']
    var['q'] = var['qo']/(const['rhow']*const['cw'])
    var['swo'] = interpolate_inp(inp['SW'], t[i])
    var['sw'] = var['swo']/(const['rhow']*const['cw'])
    inp['tb'] = interpolate_inp(inp['Tb'], t[i])
    #var['f1'] = 1 - 0.28*exp(-71.5*const['d'])-0.27*exp(-2.8*const['d'])-0.45*exp(-0.07*const['d'])
    var['f1'] = 1 - 0.27*math.exp(-2.8*const['d']) - 0.45*math.exp(-0.07*const['d'])
    res['dtc'] = 0
    var['alw'] = 1e-5*max(inp['tb'],1)
    var['con4']=16.*const['g']*var['alw']*const['znuw']**3/const['zkw']**2
    inp['us'] = interpolate_inp(inp['U_str'], t[i])
    var['usw'] = math.sqrt(const['rho']/const['rhow'])*inp['us']
    var['con5'] = var['con4']/var['usw']**4
    var['q2'] = max(1/(const['rhow']*const['cw']), - var['q'])
    var['zlan'] = 6/ (1+(var['con5']*var['q2'])**0.75)**0.333
    var['dep'] = var['zlan']*const['znuw']/var['usw']
    var['fs'] = 0.065 + 11*var['dep'] - (6.6e-5/var['dep'])*(1-math.exp(-var['dep']/8e-4))
    var['fs'] = max(var['fs'], 0.01)
    res['dtc'] = var['dep']*(var['q']+var['sw']*var['fs']) /const['zkw']
    res['dtc'] = min(res['dtc'], 0)
    res['dtw'] = 0
    var['alw']=1.e-5*max(inp['tb'],1.)
    var['con1'] = math.sqrt(5*const['d']*const['g']*var['alw']/const['an'])
    var['con2'] = const['zk']*const['g']*var['alw']
    var['qn'] = var['q']+var['sw']*var['f1']
    var['usw'] = math.sqrt(const['rho']/const['rhow'])*inp['us']
    if i !=0: var['dtwo']=res['Tw'][i-1]
    else:     var['dtwo']=0
    if(var['dtwo']>0 and var['qn']<0):
        var['qn1'] = math.sqrt(var['dtwo'])*var['usw']**2/var['con1']
        var['qn'] = max(var['qn'], var['qn1'])
    var['zeta'] = const['d']*var['con2']*var['qn']/ var['usw']**3
    if(var['zeta']>0):
        var['phi'] = 1 + 5*var['zeta']
    else:
        var['phi'] = 1/math.sqrt(1 - 16*var['zeta'])
    var['con3'] = const['zk']*var['usw']/(const['d']*var['phi'])
    res['dtw'] = (var['dtwo'] + (const['an']+1)/const['an'] * (var['q']+var['sw']*var['f1']) * delt/const['d']) / (1 + (const['an']+1)*var['con3']*delt)
    res['Tw'].append(max(0, res['dtw']))
    res['Tc'].append(res['dtc'])
    res['ts'] = inp['tb'] + res['dtw'] + res['dtc']
    res['Ts'].append(res['ts'])

#%%
fnTs = interp1d(t, res['Ts'], kind='cubic')
fnTc = interp1d(t, res['Tc'], kind='cubic')
fnTw = interp1d(t, res['Tw'], kind='cubic')
Ts = fnTs(np.linspace(t[0],t[-1], num=len(inp['LH'])))
Tc = fnTc(np.linspace(t[0],t[-1], num=len(inp['LH'])))
Tw = fnTw(np.linspace(t[0],t[-1], num=len(inp['LH'])))

#%%Post-processing
save_results = 0
get_T_full = 1

#Save the result as a DF and save it as .dat if specified
dfT = pd.DataFrame(Ts, columns=['Ts'], index=dfLH.index)
dfT['Tc'] = Tc
dfT['Tw'] = Tw
if get_T_full == 1:
    dfT_full = pd.DataFrame(res['Ts'], columns=['Ts'])
    dfT_full['Tc'] = res['Tc']
    dfT_full['Tw'] = res['Tw']
    dfT_full['AbsTime'] = t
    dfT_full = dfT_full.set_index('AbsTime')
if save_results == 1:
    T_dat_path = r'..\results\T_' + time.strftime("%Y%m%d%H%M%S") + '.dat'
    dfT.to_csv(T_dat_path, sep = ' ')
    del T_dat_path
    if get_T_full == 1:
        T_full_dat_path = r'..\results\T_full_' + time.strftime("%Y%m%d%H%M%S") + '.dat'
        dfT_full.to_csv(T_full_dat_path, sep = ' ')
        del T_full_dat_path

#Plot and save figures as png if specified
fig1 = plt.figure()
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
ax.set_ylabel('Skin Temperature in \N{DEGREE SIGN}C')
dfT['Ts'].plot(figsize=(20,6), ax=ax)
if save_results == 1:
    png_path = r'..\results\Ts_' + time.strftime("%Y%m%d%H%M%S") + '.png'
    plt.savefig(png_path)
    del png_path

if get_T_full == 1:
    fig2 = plt.figure()
    fig2 = dfT_full.plot(figsize=(20,6))
    if save_results == 1:
        png_path_full = r'..\results\Ts_full_' + time.strftime("%Y%m%d%H%M%S") + '.png'
        plt.savefig(png_path_full)
    del png_path_full

del save_results, get_T_full

#%% Clearing unwanted variables
del_inp_df = 1
if del_inp_df == 1:
    del dfLH, dfLW, dfSH, dfSW, dfU_str, dfTb
del del_inp_df, ax, fig1, fig2, i, t, delt, n_step, start_time, end_time , Ts, Tc, Tw