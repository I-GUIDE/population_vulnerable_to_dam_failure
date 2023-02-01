import pandas as pd
import geopandas as gpd
import os
import requests
import esda
import libpysal
import multiprocessing as mp
import itertools
import pygeos
import subprocess
import json
import sys


PROCESSORS = 4
cwd = os.getcwd()

scenarios = {'loadCondition': 'MH', 'breachCondition': 'F'}
input_dir = f'NID_FIM_{scenarios["loadCondition"]}_{scenarios["breachCondition"]}'
output_dir = f'{scenarios["loadCondition"]}_{scenarios["breachCondition"]}_Results'
API_Key = 'fbcac1c2cc26d853b42c4674adf905e742d1cb2b' # Census api key


# Find the list of dams in the input folder
fed_dams = pd.read_csv('./nid_available_scenario.csv')
fed_dams = fed_dams.loc[fed_dams[f'{scenarios["loadCondition"]}_{scenarios["breachCondition"]}_size'] > 0]
dois = fed_dams['ID'].to_list()
dois = dois[n*20:(n+1)*20]

# print(sys.argv)

# n = int(sys.argv[1])

# print(f"from {n*20} to {(n+1)*20}")
# print(dois[n*20:(n+1)*20])

print("END")