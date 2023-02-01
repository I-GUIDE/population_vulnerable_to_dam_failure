#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from geoedfframework.utils.GeoEDFError import GeoEDFError
# from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import multiprocessing as mp
import requests
import sys
import json
import urllib
import urllib.request
import pandas as pd
import itertools
import os


""" Module for implementing the Dam flood inundation map input connector plugin. 
    This module will implement the get() method required for all input plugins.
"""

class DamFIMInput():

    # no optional params yet, but keep around for future extension
    # a comma separated list of scenarios is required
    __optional_params = ['target_path']
    __required_params = ['dam_id','scenarios']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFInput super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise AttributeError  # TODO: uncomment the line below and remove this one
                # raise GeoEDFError('Required parameter %s for DamFIMInput not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # class super class init
        super().__init__()

    # each Input plugin needs to implement this method
    # if error, raise exception; if not, return True
    def get(self):

        # user provided scenarios to download
        user_scenarios = self.scenarios.split(',')
        
        # loop through scenarios available for this dam and download those that match the provided
        # scenario names
        r = requests.get("https://fim.sec.usace.army.mil/ci/fim/getEAPLayers?id=" + self.dam_id)
        dam_scenarios = json.loads(r.content)
        for scenario in dam_scenarios:
            for user_scenario in user_scenarios:
                if user_scenario in scenario['displayName']:
                    # then download
                    link = "https://fim.sec.usace.army.mil/ci/download/start?LAYERID="\
                    + str(scenario["layerId"])\
                    + "&type=s3&RASTER_INFO_ID=" + str(scenario["rasterInfoID"])\
                    + "&TABLE=FLOOD_DEPTH&TABLE_ID=" + str(scenario["floodDepthID"])

                    #construct filename out of load and breach condition
                    fileName = '%s/%s_%s_%s.tiff' % (self.target_path,scenario['loadCondition'],scenario['breachCondition'],self.dam_id)
                    # download file
                    try:
                        file = urllib.request.urlretrieve(link, fileName)
                        print(link)
                    except urllib.error.HTTPError as err:
                        print("DamFIMInput for %s   - HTTPError" % self.dam_id)
                    except requests.exceptions.ConnectionError as err:
                        print("DamFIMInput for %s   - ConnectionError" % self.dam_id)
                    except requests.exceptions.Timeout:
                        print("DamFIMInput for %s   - Timeout" % self.dam_id)
                    except requests.exceptions.TooManyRedirects:
                        print("DamFIMInput for %s   - TooManyRedirects" % self.dam_id)
                    except requests.exceptions.RequestException as e:
                        print("DamFIMInput for %s   - Error" % self.dam_id)
        return True


def DamFIMInput_unpacker(dam_id, scenarios, target_path):

    print("Downloading %s" % dam_id)

    return DamFIMInput(dam_id, scenarios, target_path).get()


scenarios = {'loadCondition': 'MH', 'breachCondition': 'F'}
output_path = f'NID_FIM_{scenarios["loadCondition"]}_{scenarios["breachCondition"]}'
cwd = os.getcwd()

# Find the list of dams that have MH breach scenarios
# fed_dams = requests.get('https://fim.sec.usace.army.mil/ci/fim/getAllEAPStructure').json()
# fd_df = pd.DataFrame(fed_dams)
fd_df = pd.read_csv('./nid_available_scenario.csv')
fd_df = fd_df[fd_df[f"{scenarios['loadCondition']} Breach"] == True]
dois = fd_df['ID'].to_list()
print(f"Number of dams to download: {len(dois)}")
# print(os.path.join(cwd, output_path))


for doi in dois:
    if os.path.exists(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_{doi}.tiff")):
        # print("Skipping %s" % doi)
        continue
        
    print("Downloading %s" % doi)
    DamFIMInput(dam_id=doi,scenarios=f"{scenarios['loadCondition']} Breach",target_path=os.path.join(cwd, output_path)).get()

