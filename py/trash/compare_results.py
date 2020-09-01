#%% Compare two results
import pandas as pd
import matplotlib.pyplot as plt
import operator

df1 = pd.read_csv('1.dat', sep="\s+", names=['Datetime','dTdt'], skiprows=1)
df2 = pd.read_csv('2.dat', sep="\s+", names=['Datetime','dTdt'], skiprows=1)

l=list(map(operator.sub, df1['dTdt'].tolist(), df2['dTdt'].tolist()))
plt.plot(l)

#%% Compare nlargest from results to data
# LHtop20 = dfLH.nlargest(20, 'LH').index.values
# dfLH.iloc[LHtop20]
# SHtop20 = dfSH.nlargest(20, 'SH').index.values
# dfSH.iloc[SHtop20]
# SWtop20 = dfSW.nlargest(20, 'SW').index.values
# dfSW.iloc[SWtop20]
# LWtop20 = dfLW.nlargest(20, 'LW').index.values
# dfLW.iloc[LWtop20]
# uwtop20 = dfuwstr.nlargest(20, 'uw_str').index.values
# dfuwstr.iloc[uwtop20]