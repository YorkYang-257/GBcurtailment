#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:57:05 2024

@author: york
"""

import os

# Define the directory path and script name
directory_path = 'psst'
script_name = "-m psst.cli scuc --data GeneratorInput.dat --solver 'cbc'"
script_name2 = "-m psst.cli sced --uc xfertoames.dat --data GeneratorInput.dat --solver 'cbc'"
script_name3 = "-m psst.cli scuced --fd ForecastDemand.dat --rd RealDemand.dat --gi GeneratorInput.dat --solver 'cbc'"
script_name4 = "-m psst.cli scuced --fd fd_con.dat --rd rd_con.dat --gi gtest.dat --solver 'cbc'"

# Create the command string
command = f'cd {directory_path} && python {script_name}'
command2 = f'cd {directory_path} && python {script_name4}'

# Execute the command
#for i in range(10):
os.system(command2)
