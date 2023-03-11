import pandas as pd
import geopandas as gpd
import os
import requests
import multiprocessing as mp
import itertools

def call_census_table(state_list, table_name, key):
    
    result_df = pd.DataFrame()
    
    # querying at census tract level
    for state in state_list:
        if table_name.startswith('DP'):
            address = f'https://api.census.gov/data/2020/acs/acs5/profile?get=NAME,{table_name}&for=tract:*&in=state:{state}&in=county:*'
        elif table_name.startswith('S'):
            address = f'https://api.census.gov/data/2020/acs/acs5/subject?get=NAME,{table_name}&for=tract:*&in=state:{state}&in=county:*'
        elif table_name.startswith('B'):
            address = f'https://api.census.gov/data/2020/acs/acs5?get=NAME,{table_name}&for=tract:*&in=state:{state}&in=county:*'
        else:
            raise AttributeError('Proper Table Name Is Needed.')
            
        # print(state, table_name)    
        response = requests.get(f'{address}&key={key}').json()
        result_ = pd.DataFrame(response)
        
        result_.columns = response[0]
        result_.drop(0, axis=0, inplace=True)
        
        result_df = pd.concat([result_, result_df]).reset_index(drop=True)
        
    # result_df = result_df.rename(columns={'GEO_ID':'GEOID_T'})
    result_df['GEOID_T'] = result_df.apply(lambda x: x['state'] + x['county'] + x['tract'], axis=1)
    result_df[table_name] = result_df[table_name].astype(float)
        
    return result_df[['GEOID_T', table_name]]


def census_data_retrieval(attr, state_list, tract, census_dic, API_Key, data_dir):
    attr_df = pd.DataFrame({'GEOID_T':tract['GEOID'].unique().tolist()})

    print(attr)
    cols = list(census_dic.keys())
    cols.append('GEOID_T')

    if type(census_dic[attr]) == str:
        temp_table = call_census_table(state_list, census_dic[attr], API_Key)
        attr_df = attr_df.merge(temp_table, on='GEOID_T')
        attr_df = attr_df.rename(columns={census_dic[attr]: attr})
    else:
        for table in census_dic[attr][0]: # Retrieve numerator variables
            temp_table = call_census_table(state_list, table, API_Key)
            attr_df = attr_df.merge(temp_table, on='GEOID_T')

        temp_table = call_census_table(state_list, census_dic[attr][1], API_Key) # Retrieve denominator variable
        attr_df = attr_df.merge(temp_table, on='GEOID_T')

        # Calculate the ratio of each variable
        attr_df[attr] = attr_df[census_dic[attr][0]].sum(axis=1) / attr_df[census_dic[attr][1]] * 100

    # Remove intermediate columns used for SVI related census calculation
    attr_df = attr_df[attr_df.columns.intersection(cols)]

    # attr_df.to_csv(os.path.join(data_dir, 'census_geometry', f'census_data_{attr}.csv'))

    return attr_df


def census_data_retrieval_unpacker(args):
    return census_data_retrieval(*args)

if __name__ == "__main__":

    census_dic = {
                    "EP_POV150" : [['S1701_C01_040E'], 'S1701_C01_001E'],
                    "EP_UNEMP"  : 'DP03_0009PE',
                    "EP_HBURD"  : [['S2503_C01_028E', 'S2503_C01_032E', 'S2503_C01_036E', 'S2503_C01_040E'], 
                                'S2503_C01_001E'],
                    "EP_NOHSDP" : 'S0601_C01_033E',
                    "EP_UNINSUR" : 'S2701_C05_001E',
                    "EP_AGE65" : 'S0101_C02_030E',
                    "EP_AGE17" : [['B09001_001E'], 
                                'S0601_C01_001E'],
                    "EP_DISABL" : 'DP02_0072PE',
                    "EP_SNGPNT" : [['B11012_010E', 'B11012_015E'], 'DP02_0001E'],
                    "EP_LIMENG" : [['B16005_007E', 'B16005_008E', 'B16005_012E', 'B16005_013E', 'B16005_017E', 'B16005_018E', 
                                    'B16005_022E', 'B16005_023E', 'B16005_029E', 'B16005_030E', 'B16005_034E', 'B16005_035E',
                                    'B16005_039E', 'B16005_040E', 'B16005_044E', 'B16005_045E'], 
                                'B16005_001E'],
                    "EP_MINRTY" : [['DP05_0071E', 'DP05_0078E', 'DP05_0079E', 'DP05_0080E', 
                                    'DP05_0081E', 'DP05_0082E', 'DP05_0083E'],
                                'S0601_C01_001E'],
                    "EP_MUNIT" : [['DP04_0012E', 'DP04_0013E'], 
                                'DP04_0001E'],
                    "EP_MOBILE" : 'DP04_0014PE',
                    "EP_CROWD" : [['DP04_0078E', 'DP04_0079E'], 
                                'DP04_0002E'],
                    "EP_NOVEH" : 'DP04_0058PE',
                    "EP_GROUPQ": [['B26001_001E'], 
                                'S0601_C01_001E'],
    }

    API_Key = '5ad4c26135eaa6a049525767607eecd39e19d237' # Census api key
    PROCESSORS = 4

    data_dir = os.getcwd()
    # data_dir = '/anvil/projects/x-cis220065/x-cybergis/compute/Aging_Dams'

    tract = gpd.read_file(os.path.join(data_dir, 'census_geometry', 'census_tract_from_api.geojson'))

    state_lookup = pd.read_csv(os.path.join(data_dir, 'census_geometry', 'state_lookup.csv'))
    state_lookup = state_lookup.loc[state_lookup['ContiguousUS'] == 1]
    state_lookup['FIPS'] = state_lookup['FIPS'].astype(str)
    state_list = list(state_lookup.apply(lambda x:x['FIPS'] if len(x['FIPS']) == 2 else '0' + x['FIPS'], axis=1))

    pool = mp.Pool(PROCESSORS)
    results = pool.map(census_data_retrieval_unpacker,
                            zip(
                                list(census_dic.keys()),
                                itertools.repeat(state_list),
                                itertools.repeat(tract),
                                itertools.repeat(census_dic),
                                itertools.repeat(API_Key),
                                itertools.repeat(data_dir)
                                )
                            )

    pool.close()

    census_df = pd.DataFrame({'GEOID_T':tract['GEOID'].unique().tolist()})

    for result in results:
        census_df = census_df.merge(result, on='GEOID_T')
        
    census_df['GEOID_T'] = census_df['GEOID_T'].astype(str)

    census_df.to_csv(os.path.join(data_dir, 'census_geometry', 'census_data.csv'), index=False)
    census_df.to_json(os.path.join(data_dir, 'census_geometry', 'census_data.json'))


