#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:18:34 2024

@author: york
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Demand=pd.read_csv('demanddata_2022.csv')
filtered_demand = Demand.iloc[::2]
filtered_demand=filtered_demand.reset_index(drop=True)
EW_demand=np.array(filtered_demand.loc[:,'ENGLAND_WALES_DEMAND'])
S_demand=np.array(filtered_demand.loc[:,'ND'])-np.array(filtered_demand.loc[:,'ENGLAND_WALES_DEMAND'])
rd=EW_demand+S_demand

forecast_demands0 = pd.read_csv(r"2022fd.csv")
forecast_demands= np.array(forecast_demands0).flatten()[:17520]
fd=forecast_demands[::2]

fg_vre=np.load('fg_vre.npy')
rg_vre=np.load('rg_vre.npy')

fd_con=fd-fg_vre
fd_con[fd_con<0]=0
rd_con=np.zeros(8760)

for i in range(len(rd)):
    if rd[i]>fd[i]:
        if rg_vre[i]>fg_vre[i]:
            rd_con[i]=rd[i]-rg_vre[i]
            if rd_con[i]<fd_con[i]: rd_con[i]=fd_con[i]
        else:
            rd_con[i]=rd[i]-rg_vre[i]
    else:
        if rg_vre[i]>fg_vre[i]:
            rd_con[i]=rd[i]-fg_vre[i]
        else:
            rd_con[i]=rd[i]-rg_vre[i]

with open('fd_con_tc.dat','w') as f:
    f.write('param: Demand :=\n')
    for i in range(len(rd)):
        f.write(f'Bus1 {i+1} {fd_con[i]+400}\n') 
    f.write(';')

rd_con[rd_con<0]=0
with open('rd_con.dat','w') as f:
    f.write('param: Demand :=\n')
    for i in range(len(rd)):
        f.write(f'Bus1 {i+1} {rd_con[i]}\n') 
    f.write(';')

        
        
        
        
        
        
        
        