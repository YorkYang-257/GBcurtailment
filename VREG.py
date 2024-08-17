#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:15:27 2024

@author: york
"""


import xarray as xr

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from netCDF4 import Dataset
from bng_latlon import OSGB36toWGS84
import geopandas as gpd
from shapely.geometry import Point
import pvlib
from pvlib import  irradiance


"Renewable Energy Data"
RE = pd.read_csv('repd-q1-2024.csv', encoding='ISO-8859-1')
# Changing BNG coordinates into latitude and longitude
def location(coordinates):
    x=list(OSGB36toWGS84(coordinates[0], coordinates[1]))
    latitude ,longitude=x
    return latitude, longitude

"Demand data"
Demand=pd.read_csv('demanddata_2022.csv')
filtered_demand = Demand.iloc[::2]
filtered_demand=filtered_demand.reset_index(drop=True)
EW_demand=np.array(filtered_demand.loc[:,'ENGLAND_WALES_DEMAND'])
S_demand=np.array(filtered_demand.loc[:,'ND'])-np.array(filtered_demand.loc[:,'ENGLAND_WALES_DEMAND'])

"Temperature"
tf=xr.open_dataset(r'/Users/york/Downloads/to_boao/temp.grib')
tempf=tf['t2m']
def getT(coordinates):
    latitude,longitude = location(coordinates)
    tp=(tempf.sel(latitude=latitude, longitude=longitude, method='nearest')).to_pandas()
    ta=tp.values
    return ta

"Wind Power"
# Wind speed at specific location
filein = r'/Users/york/Downloads/to_boao/2022wind100m.grib'
ds = xr.open_dataset(filein)
maxtime = '2022-12-31T23:00:00'
mintime = '2022-01-01T00:00:00'
ds = ds.sel(time=slice(mintime, maxtime))
pic_ds = ds**2
new_dataset = xr.Dataset(data_vars={"energy": pic_ds['u100']+pic_ds['v100']})
def acm_energy(coordinates):
    latitude,longitude = location(coordinates)
    
    energy_frame=(new_dataset.sel(latitude=latitude, longitude=longitude, method='nearest')).to_pandas()
    wind_limit = np.array(energy_frame['energy']).flatten()
    wind_speed = np.sqrt(wind_limit)
    "forecast wind"
    # noise = np.random.normal(-0.152, 1.46, wind_speed.shape)
    # wind_speed = wind_speed+noise
    return wind_speed

# Define wind turbine class
class WindTurbine:
    def __init__(self, Technology_type, location, Turbine_capacity, No_Turbine):
        if Technology_type=='Wind Onshore':
            self.RatedSpeed = 14
            self.cutin=3
            self.cutout=25
        else:
            self.RatedSpeed = 14
            self.cutin =  3
            self.cutout = 25
        self.location = location
        if No_Turbine==0:
            No_Turbine= int(float(Turbine_capacity)/0.5)
        self.RatedPower = float(Turbine_capacity)/int(No_Turbine)
        self.No_Turbine = int(No_Turbine)
        self.type= Technology_type
        #self.RatedSpeed = 14

#Calculating Wind turbine output with specific wind speed
def wind_power(WT,Wind_speed):
    RatedSpeed=WT.RatedSpeed
    RatedPower=WT.RatedPower
    if WT.type=='Wind Onshore':
        Wind_speed=Wind_speed*(0.8)**0.142
    if Wind_speed <= WT.cutin:
        result = 0
    elif Wind_speed > WT.cutin  and Wind_speed <= RatedSpeed:
        result = Wind_speed**3 * RatedPower / RatedSpeed**3
    elif Wind_speed > RatedSpeed and Wind_speed<= WT.cutout:
        result = RatedPower
    else:
        result = 0
    result = result*WT.No_Turbine
    return result

# Filtering data with only wind energy and currently operational
RE_Wind = RE[(RE['Technology Type']=='Wind Offshore') | (RE['Technology Type']=='Wind Onshore') ]
RE_Wind = RE_Wind[RE_Wind['Development Status']=='Operational']
# RE_Wind = RE_Wind[(RE_Wind['Development Status (short)']=='Operational')|(RE_Wind['Development Status (short)']=='Under Construction')|
#                   (RE_Wind['Development Status (short)']=='Finished')|(RE_Wind['Development Status (short)']=='Revised')|
#                   (RE_Wind['Development Status (short)']=='Application Submitted')|(RE_Wind['Development Status (short)']=='Awaiting Construction ')|
#                   (RE_Wind['Development Status (short)']=='Appeal Lodged')]
RE_Wind = RE_Wind[RE_Wind['Region']!='Northern Ireland']

# Creating an array with WindTurbine objects
WindTurbines=[]
for row in RE_Wind.itertuples():
    try:
        if float(row._9)>0:
            if float(row._17)>0:
                WindTurbines.append(WindTurbine(row._6, [int(row._29),int(row._30)], row._9, row._17))
            else:
                WindTurbines.append(WindTurbine(row._6, [int(row._29),int(row._30)], row._9, 0))
    except ValueError:
        None


# Calculating Wind power and store it with its location
# Wind_PowerAndLocation = []
# for WT in WindTurbines:
#     hour=6666
#     Wind_speed = acm_energy(WT.location)
#     power = wind_power(WT, Wind_speed[hour])
#     Wind_PowerAndLocation.append([power,WT.location])

# Sum the power in different distribution region
def Power_region(PowerAndLocation):
    "GB Distribution Network Operator license area"
    global gdf
    gdf = gpd.read_file('gb-dno-license-areas-20240503-as-geojson.geojson')
    gdf = gdf.set_index("Area")
    #Power=np.zeros(len(gdf.index))
    Power = [np.zeros(8760) for _ in gdf.index]
    Power = [[array] for array in Power]
    new_df = pd.DataFrame(Power, index=gdf.index, columns=['Power'])
    gdf["boundary"] = gdf.boundary
    n=0
    for i in PowerAndLocation:
        point=Point(i[1][0],i[1][1])
        distance=np.zeros(len(gdf.index))
        iloc=0
        for j in gdf.index:
            if iloc<len(gdf.index):
                region=gdf.loc[j]
                is_within = point.within(region["geometry"])
                if is_within:   
                    new_df.loc[j,'Power']=new_df.loc[j,'Power']+i[0]
                    iloc=len(gdf.index)
                    n=n+1
                else: 
                    dis=point.distance(region["boundary"])
                    distance[iloc]=dis
                iloc=iloc+1
        if iloc<len(gdf.index)+1:
            iloc=np.argmin(distance)
            j=gdf.index[iloc]
            new_df.loc[j,'Power']=new_df.loc[j,'Power']+i[0]
            n=n+1
    return new_df
        
        
        
        
"Solar Power"
# Read solar radiance at specific loaction
dataset = Dataset(r'/Users/york/Downloads/to_boao/2022solar.nc')
#print(dataset)

lons = dataset.variables['longitude'][:]
lats = dataset.variables['latitude'][:]
times = dataset.variables['time']
def acm_solar(coordinates):

    #print(lons, lats, times)
    latitude,longitude = location(coordinates)
    given_lon = longitude  # 经度值
    given_lat = latitude  # 纬度值

    # Define your location (latitude, longitude, timezone)
    site = pvlib.location.Location(latitude=given_lat, longitude=given_lon, tz='Europe/London')

    # Specify the times for your calculation
    times = pd.date_range(start='2022-01-01 00:00:00', end='2022-12-31 23:00:00', freq='h', tz=site.tz)
    panel_azimuth = 180  # Facing south
    panel_tilt = 45      # 45 degrees tilt
    clear_sky = site.get_clearsky(times)
    solar_position = site.get_solarposition(times)
    # 找到最接近给定经纬度的网格点的索引
    lon_idx = np.abs(lons - given_lon).argmin()
    lat_idx = np.abs(lats - given_lat).argmin()

    ssrd_data = dataset.variables['ssrd'][:, lat_idx, lon_idx]/3600
    
    poa_irradiance = irradiance.get_total_irradiance(
    surface_tilt=panel_tilt,
    surface_azimuth=panel_azimuth,
    dni=clear_sky['dni'],
    ghi=ssrd_data,
    dhi=clear_sky['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth'])
    ghi=poa_irradiance['poa_global'].values
    "forecast irradiance"
    # noise = np.random.normal(6.23, 48.41, ghi.shape)
    # ghi = ghi+noise
    return ghi

# Define solar panel class
class SolarPanel:
    def __init__(self, location, Panel_capacity):
        # if Technology_type='Wind Onshore'：
        #     self.type = 'Onshore'
        # else:
        #     self.type = 'Offshore'
        self.location = location
        self.RatedPower = float(Panel_capacity)
        self.No_Pannel = float(Panel_capacity)*1111
        self.Ratedirridiance=1000

# Calculating Solar Power
def Solar_Power(SP,SI,temp):
    if SI<SP.Ratedirridiance and SI>0:
        k1=-0.017162
        k2=-0.040289
        k3=-0.004681
        k4=0.000148
        k5=0.000169
        k6=0.000005
        G_prime=SI/SP.Ratedirridiance
        #print(G_prime)
        T_prime=temp+0.05*SI-298.15
        term1 = k1 * np.log(G_prime)
        term2 = k2 * (np.log(G_prime)**2)
        term3 = T_prime * (k3 + k4 * np.log(G_prime)+k5 * (np.log(G_prime)**2))
        term4 = 0
        term5 = k6 * (T_prime**2)
        n_rel = 1 + term1 + term2 + term3 + term4 + term5
        power=SP.RatedPower*SI/SP.Ratedirridiance*n_rel
        if n_rel<0: power=0
    else:
        if SI>SP.RatedPower:
             power=SP.RatedPower
        else:
             power=0
    return power

# Filtering data with only solar energy and currently operational
RE_Solar = RE[RE['Technology Type']=='Solar Photovoltaics']
RE_Solar = RE_Solar[RE_Solar['Development Status']=='Operational']
# RE_Solar = RE_Solar[(RE_Solar['Development Status (short)']=='Operational')|(RE_Solar['Development Status (short)']=='Under Construction')|
#                   (RE_Solar['Development Status (short)']=='Finished')|(RE_Solar['Development Status (short)']=='Revised')|
#                   (RE_Solar['Development Status (short)']=='Application Submitted')|(RE_Solar['Development Status (short)']=='Awaiting Construction ')|
#                   (RE_Solar['Development Status (short)']=='Appeal Lodged')]
RE_Solar = RE_Solar[RE_Solar['Region']!='Northern Ireland']

# Creating an array with SolarPanel objects
SolarPanels=[]
for row in RE_Solar.itertuples():
    if float(row._9)>0 and float(row._29)>0:
        SolarPanels.append(SolarPanel([int(row._29),int(row._30)], row._9))

# Calculating Solar Panel output with specific irradiation
# Solar_PowerAndLocation = []
# for SP in SolarPanels:
#     hour=6666
#     Solar_irradiance = acm_solar(SP.location)/3600
#     power = Solar_Power(SP, Solar_irradiance[hour])/1000000
#     Solar_PowerAndLocation.append([power,SP.location])
# data=Power_region(Wind_PowerAndLocation)
# data2 = Power_region(Solar_PowerAndLocation)

# Solar_Periods=[]
# for hour in range(720,768):
#     Solar_PowerAndLocation = []
#     for SP in SolarPanels:
#         #hour=4812
#         Solar_irradiance = acm_solar(SP.location)/3600
#         power = Solar_Power(SP, Solar_irradiance[hour])/1000000
#         Solar_PowerAndLocation.append([power,SP.location])
#     print(hour)
#     data2= Power_region(Solar_PowerAndLocation)
#     Solar_Periods.append(data2)
# Solar_Periods=pd.concat(Solar_Periods,axis=1)

"optimisation of code"
Solar_Periods=[]
Solar_PowerAndLocation = []
for SP in SolarPanels:
    Solar_irradiance = acm_solar(SP.location)
    temp=getT(SP.location)
    power=[]
    for hour in range(8760):
        power.append(Solar_Power(SP, Solar_irradiance[hour],temp[hour]))
    power=np.array(power)
    Solar_PowerAndLocation.append([power,SP.location])
    #print(1)
Solar_Periods= Power_region(Solar_PowerAndLocation)



# Wind_Periods=[]
# for hour in range(720,768):
#     Wind_PowerAndLocation = []
#     for WT in WindTurbines:
#         Wind_speed = acm_energy(WT.location)
#         power = wind_power(WT, Wind_speed[hour])
#         Wind_PowerAndLocation.append([power,WT.location])    
#     print(hour)
#     data=Power_region(Wind_PowerAndLocation)
#     Wind_Periods.append(data)
# Wind_Periods=pd.concat(Wind_Periods,axis=1)

Wind_Periods=[]
Wind_PowerAndLocation=[]
for WT in WindTurbines:
    Wind_speed = acm_energy(WT.location)
    power=[]
    for hour in range(8760):
        power.append(wind_power(WT, Wind_speed[hour]))
    power=np.array(power)
    Wind_PowerAndLocation.append([power,WT.location])
Wind_Periods = Power_region(Wind_PowerAndLocation)


area_mapping = {
    'East England': 'England_Wales', 'East Midlands': 'England_Wales', 'London': 'England_Wales',
    'North Wales, Merseyside and Cheshire': 'England_Wales', 'Southern England': 'England_Wales', 'West Midlands': 'England_Wales', 'North East England': 'England_Wales',
    'North West England': 'England_Wales', 'South East England': 'England_Wales', 'South Wales': 'England_Wales',
    'South West England': 'England_Wales', 'Yorkshire': 'England_Wales', 'North Scotland': 'Scotland', 'South and Central Scotland': 'Scotland'
}
Solar_Periods['Major Area']=Solar_Periods.index.map(area_mapping)
result_solar_df = Solar_Periods.groupby('Major Area').sum()
Wind_Periods['Major Area']=Wind_Periods.index.map(area_mapping)
result_wind_df = Wind_Periods.groupby('Major Area').sum()


"plot graph"
power_E_wind = result_wind_df.loc['England_Wales', 'Power']
power_S_wind = result_wind_df.loc['Scotland', 'Power']
power_E_solar = result_solar_df.loc['England_Wales', 'Power']
power_S_solar = result_solar_df.loc['Scotland', 'Power']
x_values = np.arange(8760)  # Array from 0 to 8759

# Plot
# Create the figure and axis
plt.figure(figsize=(12, 6))

# Stacked area plot
plt.stackplot(x_values, power_S_wind+power_E_wind, power_S_solar+power_E_solar, labels=['Wind', 'Solar'], alpha=0.5)

# Line plot on the same axes
plt.plot(x_values, S_demand+EW_demand, label='Demand', color='blue', linestyle='--')

# Adding labels and title
plt.title('GB demand and generation')
plt.xlabel('Hour of the Year')
plt.ylabel('Power Generated')
plt.legend(loc='upper right')

# Show the plot
plt.show()


