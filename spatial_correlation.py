import pandas as pd
import geopandas as gpd
import os
import requests
import esda
import libpysal
import multiprocessing as mp
import itertools


def call_census_table(state_list, table, key):
    
    result_df = pd.DataFrame()
    
    # querying at census tract level
    for state in state_list:
        if table.startswith('group'):
            address = f'https://api.census.gov/data/2020/acs/acs5?get=NAME,{table}&for=tract:*&in=state:{state}&in=county:*'
        elif table.startswith('DP'):
            address = f'https://api.census.gov/data/2020/acs/acs5/profile?get=NAME,{table}&for=tract:*&in=state:{state}&in=county:*'
        elif table.startswith('S'):
            address = f'https://api.census.gov/data/2020/acs/acs5/subject?get=NAME,{table}&for=tract:*&in=state:{state}&in=county:*'
        response = requests.get(f'{address}&key={key}').json()
        result_ = pd.DataFrame(response)
        
        result_.columns = response[0]
        result_.drop(0, axis=0, inplace=True)
        
        if table.startswith('group'):
            result_.drop(['NAME', 'state', 'county', 'tract'], axis=1, inplace=True) 
        else:
            result_.drop(['NAME'], axis=1, inplace=True) # When querying tract level data
        
        result_df = pd.concat([result_, result_df]).reset_index(drop=True)
        
    return result_df


def census_data_of_fim_geoid(fim_geoid_, census_name, attr_dic, API_Key):

    # List of states that is associated with the dam failure
    state_list = fim_geoid_.apply(lambda x:x['GEOID'][0:2], axis=1).unique()

    # Retrieve census data from API
    census_code = attr_dic[census_name]
    census_table = call_census_table(state_list, census_code, API_Key)
    
    # Define GEOID for census tracts
    if 'GEO_ID' in census_table.columns:
        census_table['GEO_ID'] = census_table.apply(lambda x:x['GEO_ID'][9:], axis=1)
        census_table['GEO_ID'] = census_table['GEO_ID'].astype(str)
        census_table = census_table.rename(columns={'GEO_ID': 'GEOID_tract'})
    else:
        census_table['GEOID_tract'] = census_table.apply(lambda x:x['state'] + x['county'] + x ['tract'], axis=1)
    
    # Clean census data and merge with fim_geoid data
    if census_code == 'group(B06009)': # No high school diploma: Persons (age 25+) with no high school diploma 
        census_table[census_name] = census_table.apply(lambda x:round(int(x['B06009_002E']) / int(x['B06009_001E']) * 100.0) 
                                                     if int(x['B06009_001E']) != 0 else 0, axis=1)
    elif census_code == 'group(B17001)': # Poverty: persons below poverty estimate (not available at bg level)
        census_table[census_name] = census_table.apply(lambda x:round(int(x['B17001_002E']) / int(x['B17001_001E']) * 100.0) 
                                                     if int(x['B17001_001E']) != 0 else 0, axis=1)
    elif census_code == 'group(B16005)': # Not proficient English: "NATIVITY BY LANGUAGE SPOKEN AT HOME BY ABILITY TO SPEAK ENGLISH FOR THE POPULATION 5 YEARS AND OVER"
        census_table[census_name] = census_table.apply(lambda x:round((int(x['B16005_007E']) + int(x['B16005_008E']) +
                                                                     int(x['B16005_012E']) + int(x['B16005_013E']) + 
                                                                     int(x['B16005_017E']) + int(x['B16005_018E']) + 
                                                                     int(x['B16005_022E']) + int(x['B16005_023E']) + 
                                                                     int(x['B16005_029E']) + int(x['B16005_030E']) + 
                                                                     int(x['B16005_034E']) + int(x['B16005_035E']) + 
                                                                     int(x['B16005_039E']) + int(x['B16005_040E']) + 
                                                                     int(x['B16005_044E']) + int(x['B16005_045E'])) 
                                                                    / int(x['B16005_001E']) * 100.0) 
                                                     if int(x['B16005_001E']) != 0 else 0, axis=1)
    else:
        census_table.rename(columns={census_code: census_name}, inplace=True)

    census_table[census_name] = census_table[census_name].astype(float)   
    fim_geoid_ = fim_geoid_.merge(census_table[['GEOID_tract', census_name]], on='GEOID_tract')
    
    return fim_geoid_


def calculate_bivariate_Moran_I_and_LISA(dam_id, attr_dic, fim_geoid_gdf, dams_gdf):

    print(f'Calculating Moran\'s I and LISA for {dam_id}...')
    input_cols = list(attr_dic.keys())
    input_cols.extend(['Dam_ID', 'GEOID', 'Class', 'geometry'])
    fim_geoid_local = fim_geoid_gdf.loc[fim_geoid_gdf['Dam_ID'] == dam_id, input_cols].reset_index(drop=True)
    dam_local = dams_gdf.loc[dams_gdf['ID'] == dam_id].reset_index(drop=True)
    
    # Iterate through all census variables
    for census_name, census_code in attr_dic.items():
        # Local fim_geoid
        fim_geoid_local_var = fim_geoid_local.loc[~fim_geoid_local[census_name].isna(), ['Dam_ID', 'GEOID', 'Class', census_name, 'geometry']].reset_index(drop=True)

        # TODO: investigate proportional bandwidth or Kenel window for distance decay
        # Calculate Bivaraite Moran's I & Local Moran's I
        w = libpysal.weights.Queen.from_dataframe(fim_geoid_local_var)  # Adjacency matrix (Queen case)
        bv_mi = esda.Moran_BV(fim_geoid_local_var['Class'], fim_geoid_local_var[census_name], w)
        bv_lm = esda.Moran_Local_BV(fim_geoid_local_var['Class'], fim_geoid_local_var[census_name], w, seed=17)
        
        # Enter results of Bivariate LISA into each census region
        lm_dict = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        for idx in range(fim_geoid_local_var.shape[0]):
            if bv_lm.p_sim[idx] < 0.05:
                fim_geoid_local_var.loc[idx, f'LISA_{census_name}'] = lm_dict[bv_lm.q[idx]]
            else:
                fim_geoid_local_var.loc[idx, f'LISA_{census_name}'] = 'Not_Sig'
        
        fim_geoid_local_na = fim_geoid_local.loc[fim_geoid_local[census_name].isna(), ['Dam_ID', 'GEOID', 'Class', census_name, 'geometry']]
        fim_geoid_local_na[f'LISA_{census_name}'] = 'NA'
        fim_geoid_local_var = pd.concat([fim_geoid_local_var, fim_geoid_local_na]).reset_index(drop=True)       
        fim_geoid_local = fim_geoid_local.merge(fim_geoid_local_var[['GEOID', f'LISA_{census_name}']], on='GEOID')
                
        # Enter Bivariate Moran's I result into each dam
        dam_local[f'I_{census_name}'] = bv_mi.I
        dam_local[f'pval_{census_name}'] = bv_mi.p_z_sim

    return fim_geoid_local, dam_local


def calculate_bivariate_Moran_I_and_LISA_unpacker(args):
    return calculate_bivariate_Moran_I_and_LISA(*args)



########## Main code starts here ##########

PROCESSORS = 4
scenarios = {'loadCondition': 'TAS', 'breachCondition': 'F'}
output_path = 'output'
cwd = os.getcwd()
API_Key = 'fbcac1c2cc26d853b42c4674adf905e742d1cb2b' # Census api key

# Load spatial intersection results, which is generated by `spatial_intersection_ellipse.py`
fim_geoid = gpd.read_file(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_fim_geoid.geojson"))

# Import list of dams
dams = requests.get('https://fim.sec.usace.army.mil/ci/fim/getAllEAPStructure').json()
dams = pd.DataFrame(dams)
dams = gpd.GeoDataFrame(dams, geometry=gpd.points_from_xy(dams['LON'], dams['LAT'], crs="EPSG:4326"))
dams = dams.loc[dams['ID'].isin(fim_geoid['Dam_ID'])]
print(f"Dam of interest count: {dams.shape[0]}")

# List of census data to be retrieved. 
# TODO: add more census data
census_attr_dic = {'no_hs_dip': 'group(B06009)',     # Percentage of people over 25 without high school diploma
                    'poverty': 'group(B17001)',      # Percentage of people below the poverty level
                    'unprof_eng': 'group(B16005)',   # Percentage of resident with no proficient English
                    'mobile_home': 'DP04_0014PE',    # Percentage of mobile homes estimate
                    'no_vehicle': 'DP04_0058PE',     # Percentage of housholds without vehicle available estimate
                    'unemployed': 'DP03_0009PE',   # Unemployment Rate estimate
                    'age65': 'S0101_C02_030E'        # Percentage of person aged 65 and older estimate
                    }


# Retrieve census data from API
fim_geoid['GEOID_tract'] = fim_geoid.apply(lambda x:x['GEOID'][0:11], axis=1)
for attr in census_attr_dic.keys():
    fim_geoid = census_data_of_fim_geoid(fim_geoid, attr, census_attr_dic, API_Key)
    print(f"Census Data ({attr}) is retrieved")

# Reproject fim_geoid to EPSG:5070, NAD83 / Conus Albers (meters)
fim_geoid = fim_geoid.to_crs(epsg=5070)

# Replace not valid value (e.g., -666666) from census with nan value
for attr in census_attr_dic.keys():
    fim_geoid[attr] = fim_geoid.apply(lambda x: float('nan') if x[attr] < 0 else x[attr], axis=1)

# dois = ['TX00004', 'TX00006', 'TX00018', 'TX00020']


if __name__ == "__main__":
    # Empty GeoDataFrame for storing the results
    corr_result = pd.DataFrame()
    dams_result = pd.DataFrame()
    dois = fim_geoid['Dam_ID'].unique()

    pool = mp.Pool(PROCESSORS)
    results = pool.map(calculate_bivariate_Moran_I_and_LISA_unpacker,
                            zip(dois, # list of DAM ID 
                                itertools.repeat(census_attr_dic),
                                itertools.repeat(fim_geoid),
                                itertools.repeat(dams))
                       )

    pool.close()

    # Saving results of spatial correlation analysis
    for result in results:
        corr_result = pd.concat([corr_result, result[0]]).reset_index(drop=True)
        dams_result = pd.concat([dams_result, result[1]]).reset_index(drop=True)

    lm_cols = [f'LISA_{var}' for var in census_attr_dic.keys()]
    lm_cols.append('GEOID')

    dams_result = gpd.GeoDataFrame(dams_result, geometry=gpd.points_from_xy(dams_result['LON'], dams_result['LAT'], crs="EPSG:4326"))
    fim_geoid = fim_geoid.merge(corr_result[lm_cols], on='GEOID')
    fim_geoid = fim_geoid.to_crs(epsg=4326)

    dams_result.to_csv(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_dams_result.geojson"))
    fim_geoid.to_file(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_fim_geoid_corr.geojson"))

