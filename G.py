#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:02:12 2024

@author: york
"""

import numpy as np
import pandas as pd

Glist=pd.read_csv('Generators.csv', encoding='ISO-8859-1')
Glist=Glist[['Technology','Type','Primary Fuel','InstalledCapacity (MW)','Country','X-Coordinate','Y-Coordinate']]
Glist = Glist[~Glist['Technology'].isin(['Wind', 'Solar', 'Hydro'])]
Glist = Glist[~Glist['Country'].isin(['Northern Ireland'])]
Glist = Glist[Glist['X-Coordinate']>0]

Type=Glist['Type'].unique()
Fuel=Glist['Primary Fuel'].unique()
#print(sum(Glist['InstalledCapacity (MW)']))

gl=Glist.reset_index(drop=True)

#gl['minP','UT','DT','RL','PC']=0

def MinPrule(x):
    if x=='CCGT':
        return 0.25
    elif x=='Single Cycle' or x=='Single cycle':
        return 0.5
    elif x=='Bioenergy':
        return 0.35
    elif x=='AGR' or x=='PWR':
        return 0.9
    else:
        return 0.25
gl['minP']=gl['Type'].apply(MinPrule)

def UTrule(x):
    if x=='CCGT':
        return 4
    elif x=='Single Cycle' or x=='Single cycle':
        return 1
    elif x=='Bioenergy':
        return 8
    elif x=='AGR' or x=='PWR':
        return 48
    else:
        return 8
gl['UT']=gl['Type'].apply(UTrule)

def DTrule(x):
    if x=='CCGT':
        return 7
    elif x=='Single Cycle' or x=='Single cycle':
        return 0
    elif x=='Bioenergy':
        return 14
    elif x=='AGR' or x=='PWR':
        return 24
    else:
        return 11
gl['DT']=gl['Type'].apply(DTrule)

def RLrule(x):
    if x=='Single Cycle' or x=='Single cycle' or x=='CCGT':
        return 402
    else:
        return 120
gl['RL']=gl['Type'].apply(RLrule)

def Prule(x):
    if x=='Natural Gas' or x=='Sour Gas':
        return 104.25
    elif x=='MSW' or x=='Biomass':
        return 128.55
    elif x=='Nuclear':
        return 110
    elif x=='Coal':
        return 155.35
    else:
        return 202
gl['PC']=gl['Primary Fuel'].apply(Prule)

for index,row in gl.iterrows():
    minp=round(row['InstalledCapacity (MW)']*row['minP'],2)
    maxp=row['InstalledCapacity (MW)']
    minT=row['UT']
    maxT=row['DT']
    RL=row['RL']
    SUC=round(row['PC']*minp/4,1)
    with open('test.dat','a') as f:
        f.write(f'GenCo{index+1} {minp} 1 0 0 {minp} {maxp} {minT} {maxT} {RL} {RL} {minp} {minp} {SUC} {SUC*4} 0')
        
        f.write('\n')

for index,row in gl.iterrows():

    SUC=row['PC']
    with open('test.dat','a') as f:
        f.write(f'GenCo{index+1} 0 {SUC} 0') 
                
        f.write('\n')

# for index,row in gl.iterrows():

#     with open('test.dat','a') as f:
#         f.write(f'GenCo{index+1} ') 


