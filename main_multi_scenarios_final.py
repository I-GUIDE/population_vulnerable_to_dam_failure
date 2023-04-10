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


def resample_raster(rasterfile_path, filename, target_path, rescale_factor):
    # first determine pixel size for resampling
    xres = 0
    yres = 0
    
    out = subprocess.run(["gdalinfo","-json",rasterfile_path],stdout=subprocess.PIPE)
    raster_meta = json.loads(out.stdout.decode('utf-8'))
    if 'geoTransform' in raster_meta:
        xres = raster_meta['geoTransform'][1]
        yres = raster_meta['geoTransform'][5]
        xres = xres * rescale_factor
        yres = yres * rescale_factor

    if (xres != 0) and (yres != 0):
        # resample raster
        save_path = target_path +"/"+ filename + f"_resample.tiff"
        subprocess.run(["gdalwarp","-r","bilinear","-of","GTiff","-tr",str(xres),str(yres),rasterfile_path,save_path])

        return save_path, raster_meta
    

def polygonize_fim(rasterfile_path):

    # Extract target path and filename from the given raster file path
    target_path = '/'.join(rasterfile_path.split('/')[:-1])
    filename = rasterfile_path.split("/")[-1].split(".")[-2]

    # Define paths
    resample_path = target_path +"/"+ filename + f"_resample.tiff"
    reclass_file = target_path + "/" + filename + "_reclass.tiff"
    geojson_out = "%s/%s.json" % (target_path, filename)

    for temp_path_ in [resample_path, reclass_file, geojson_out]:
        if os.path.exists(temp_path_):
            os.remove(temp_path_)

    # Resample raster file to 10-times smaller
    resample_path, raster_meta = resample_raster(rasterfile_path, filename, target_path, rescale_factor=4)

    # Reclassify raster
    '''
    water_lvl = [0, 2, 6, 15, np.inf]  # Original inundation map value (underwater in feet)
    water_lvl_recls = [-9999, 1, 2, 3, 4]
    '''
    outfile = "--outfile="+reclass_file
    no_data_val = raster_meta['bands'][0]['noDataValue']
    subprocess.run(["gdal_calc.py","-A",resample_path,outfile,f"--calc=-9999*(A<=0)+1*(A>0)",f"--NoDataValue={no_data_val}"],stdout=subprocess.PIPE)
        
    # Polygonize the reclassified raster
    subprocess.run(["gdal_polygonize.py", reclass_file, "-b", "1", geojson_out, filename, "value"])

    inund_polygons = gpd.read_file(geojson_out)

    if inund_polygons.shape[0] != 0:
        inund_polygons = inund_polygons.loc[(inund_polygons['value'] != -9999) & (inund_polygons['value'] != 0)]  # Remove pixels of null value

        # drop invalid geometries
        inund_polygons = inund_polygons.loc[inund_polygons['geometry'].is_valid, :]

        # Coverage for each class of inundation map
        inund_per_cls = inund_polygons.dissolve(by='value')
        inund_per_cls.reset_index(inplace=True)

        # remove all temp files
        os.remove(resample_path)
        os.remove(reclass_file)
        os.remove(geojson_out)

        # inundation_per_cls: GeoDataFrame 
        return inund_per_cls

    else:
        return gpd.GeoDataFrame(data={'value': 1}, index=[0], geometry=[None])

    # inundation_per_cls: GeoDataFrame 
    return inund_per_cls


def fim_multiple_scenarios(dam_id, input_dir):
    
    sce_mh = {'loadCondition': 'MH', 'breachCondition': 'F'}  # Maximun Height scenario
    sce_tas = {'loadCondition': 'TAS', 'breachCondition': 'F'}  # Top of Active Storage scenario
    sce_nh = {'loadCondition': 'NH', 'breachCondition': 'F'}  # Normal Height scenario

    # Maximun Height scenario (weight: 1)
    fim_path_mh = f"{input_dir}/NID_FIM_{sce_mh['loadCondition']}_{sce_mh['breachCondition']}/{sce_mh['loadCondition']}_{sce_mh['breachCondition']}_{dam_id}.tiff"
    fim_gdf_mh = polygonize_fim(fim_path_mh)
    fim_gdf_mh['value_mh'] = fim_gdf_mh['value'] * 1
    fim_gdf_mh.drop(columns=['value'], inplace=True)

    # Top of Active Storage scenario (weight: 1)
    fim_path_tas = f"{input_dir}/NID_FIM_{sce_tas['loadCondition']}_{sce_tas['breachCondition']}/{sce_tas['loadCondition']}_{sce_tas['breachCondition']}_{dam_id}.tiff"
    fim_gdf_tas = polygonize_fim(fim_path_tas)
    fim_gdf_tas['value_tas'] = fim_gdf_tas['value'] * 1
    fim_gdf_tas.drop(columns=['value'], inplace=True)

    # Normal Height scenario (weight: 1)
    fim_path_nh = f"{input_dir}/NID_FIM_{sce_nh['loadCondition']}_{sce_nh['breachCondition']}/{sce_nh['loadCondition']}_{sce_nh['breachCondition']}_{dam_id}.tiff"
    fim_gdf_nh = polygonize_fim(fim_path_nh)
    fim_gdf_nh['value_nh'] = fim_gdf_nh['value'] * 1
    fim_gdf_nh.drop(columns=['value'], inplace=True)

    # Find intersections of inundated area across multiple scenarios
    temp_fim_gdf = gpd.overlay(fim_gdf_nh, fim_gdf_tas, how='union')
    fim_gdf = gpd.overlay(temp_fim_gdf, fim_gdf_mh, how='union')
    fim_gdf.fillna(0, inplace=True)

    # Sum values (1: MH only, 2: MH + TAS , 3: MH + TAS + NH)
    fim_gdf['value'] = fim_gdf.apply(lambda x:x['value_mh'] + x['value_tas'] + x['value_nh'], axis=1)
    fim_gdf.drop(columns=['value_mh', 'value_tas', 'value_nh'], inplace=True)
    fim_gdf['Dam_ID'] = dam_id
        
    return fim_gdf


def state_num_related_to_fim(fim_gdf, tract_gdf):
    
    tract_geoms = pygeos.from_shapely(tract_gdf['geometry'].values)
    tract_geoms_tree = pygeos.STRtree(tract_geoms, leafsize=50)

    fim_geom_union = pygeos.from_shapely(fim_gdf['geometry'].unary_union)    
    query_intersect = tract_geoms_tree.query(fim_geom_union, predicate='intersects')
    tract_gdf = tract_gdf.loc[query_intersect]

    tract_gdf['STATE'] = tract_gdf.apply(lambda x:x['GEOID'][0:2], axis=1)
    unique_state = tract_gdf['STATE'].unique()
    
    # return type: list
    return unique_state
    

def extract_fim_geoid(dam_id, input_dir, tract_gdf):
    print(f'{dam_id}: Step 1, 1/4, Identifying associated regions for multiple scenarios')
    fim_gdf = fim_multiple_scenarios(dam_id, input_dir)

    print(f'{dam_id}: Step 1, 2/4, Search states associated')
    fim_state = state_num_related_to_fim(fim_gdf, tract_gdf)
    print(f'-- {dam_id} impacts {len(fim_state)} States, {fim_state}')

    if len(fim_state) == 1: # If only one state is associated with the inundation mapping
        block_gdf = gpd.read_file(f'{input_dir}/census_geometry/tl_2020_{fim_state[0]}_tabblock20.geojson')
    elif len(fim_state) >= 2: # If multiple states are associated with the inundation mapping
        block_gdf = pd.DataFrame()
        for state_num in fim_state:
            temp_gdf = gpd.read_file(f'{input_dir}/census_geometry/tl_2020_{state_num}_tabblock20.geojson')
            block_gdf = pd.concat([temp_gdf, block_gdf]).reset_index(drop=True)
        block_gdf = gpd.GeoDataFrame(block_gdf, geometry=block_gdf['geometry'], crs="EPSG:4326")
    else:
        raise AttributeError('NO STATE is related to Inundation Mapping')

    # Destination dataframe to save the results
    print(f"{dam_id}: Step 1, 3/4, Extracting GEOID of census blocks")
    fim_geoid_df = pd.DataFrame({'Dam_ID': pd.Series(dtype='str'),
                                'GEOID': pd.Series(dtype='str'),
                                'Class': pd.Series(dtype='str')}
                                )    

    # Create STRtree for block_gdf
    block_geoms = pygeos.from_shapely(block_gdf['geometry'].values)
    block_geoms_tree = pygeos.STRtree(block_geoms, leafsize=50)

    # Extract census tract intersecting with each class of inundation map
    for water_cls in fim_gdf['value'].unique():
        fim_geom_ = pygeos.from_shapely(fim_gdf.loc[fim_gdf['value'] == water_cls, 'geometry'].unary_union)
        query_fim_geom_ = block_geoms_tree.query(fim_geom_, predicate='intersects')
        fim_geoid_ = block_gdf.loc[query_fim_geom_]

        for geoid_ in fim_geoid_['GEOID'].to_list():
            new_row = pd.DataFrame({'Dam_ID': dam_id, 
                                    'GEOID': geoid_, 
                                    'Class': water_cls}, 
                                    index=[0]
                                    )
            fim_geoid_df = pd.concat([new_row, fim_geoid_df]).reset_index(drop=True)

    print(f"{dam_id}: Step 1, 4/4, Assigning geometry to census blocks")
    fim_geoid_gdf = fim_geoid_df.merge(block_gdf, on='GEOID')
    fim_geoid_gdf = gpd.GeoDataFrame(fim_geoid_gdf, geometry=fim_geoid_gdf['geometry'], crs='EPSG:4326')
    fim_geoid_gdf['Class'] = fim_geoid_gdf['Class'].astype(int)
    fim_geoid_gdf = fim_geoid_gdf.groupby(['Dam_ID', 'GEOID'], 
                                    group_keys=False).apply(lambda x:x.loc[x['Class'].idxmax()]
                                                            ).reset_index(drop=True)
    fim_geoid_gdf = fim_geoid_gdf.set_crs(epsg=4326)

    return fim_geoid_gdf, fim_gdf


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
            
        response = requests.get(f'{address}&key={key}').json()
        result_ = pd.DataFrame(response)
        
        result_.columns = response[0]
        result_.drop(0, axis=0, inplace=True)
        
        result_df = pd.concat([result_, result_df]).reset_index(drop=True)
        
    # result_df = result_df.rename(columns={'GEO_ID':'GEOID_T'})
    result_df['GEOID_T'] = result_df.apply(lambda x: x['state'] + x['county'] + x['tract'], axis=1)
    result_df[table_name] = result_df[table_name].astype(float)
        
    return result_df[['GEOID_T', table_name]]


def calculate_bivariate_Moran_I_and_LISA(dam_id, census_dic, fim_geoid_gdf, dams_gdf):

    input_cols = list(census_dic.keys())
    input_cols.extend(['Dam_ID', 'GEOID', 'Class', 'geometry'])
    fim_geoid_local = fim_geoid_gdf.loc[fim_geoid_gdf['Dam_ID'] == dam_id, input_cols].reset_index(drop=True)
    dam_local = dams_gdf.loc[dams_gdf['ID'] == dam_id].reset_index(drop=True)

    # Iterate through all census variables
    for census_name in census_dic.keys():
        new_col_name = census_name.split("_")[1]
        
        # Remove invalid values of census data for local fim_geoid 
        fim_geoid_local_var = fim_geoid_local.loc[(~fim_geoid_local[census_name].isna()) & (fim_geoid_local[census_name] >= 0), 
        ['Dam_ID', 'GEOID', 'Class', census_name, 'geometry']].reset_index(drop=True)
        
        # Calculate Bivaraite Moran's I & Local Moran's I with Queen's Case Contiguity
        w = libpysal.weights.Queen.from_dataframe(fim_geoid_local_var)  # Adjacency matrix (Queen case)
        bv_mi = esda.Moran_BV(fim_geoid_local_var['Class'], fim_geoid_local_var[census_name], w)          
        bv_lm = esda.Moran_Local_BV(fim_geoid_local_var['Class'], fim_geoid_local_var[census_name], w, seed=17)

        # Enter results of Bivariate LISA into each census region
        lm_dict = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        for idx in range(fim_geoid_local_var.shape[0]):
            if bv_lm.p_sim[idx] < 0.05:
                fim_geoid_local_var.loc[idx, f'LISA_{new_col_name}'] = lm_dict[bv_lm.q[idx]]
            else:
                fim_geoid_local_var.loc[idx, f'LISA_{new_col_name}'] = 'Not_Sig'

        fim_geoid_local_na = fim_geoid_local.loc[fim_geoid_local[census_name].isna(), ['Dam_ID', 'GEOID', 'Class', census_name, 'geometry']]
        fim_geoid_local_na[f'LISA_{new_col_name}'] = 'NA'
        fim_geoid_local_var = pd.concat([fim_geoid_local_var, fim_geoid_local_na]).reset_index(drop=True)       
        fim_geoid_local = fim_geoid_local.merge(fim_geoid_local_var[['GEOID', f'LISA_{new_col_name}']], on='GEOID')

        # Enter Bivariate Moran's I result into each dam
        dam_local[f'MI_{new_col_name}'] = bv_mi.I
        dam_local[f'pval_{new_col_name}'] = bv_mi.p_z_sim

    return dam_local, fim_geoid_local


def spatial_correlation(dam_id, fd_gdf, fim_geoid_gdf, census_dic, census_df):
    
    # Merging census data to FIM geoid
    print(f"{dam_id}: Step 2, 1/2, Merging census data to FIM geoid")
    fim_geoid_gdf['GEOID_T'] = fim_geoid_gdf.apply(lambda x:x['GEOID'][0:11], axis=1)

    # Merge census data with fim_geoid_gdf
    fim_geoid_gdf = fim_geoid_gdf.merge(census_df, on='GEOID_T')

    # Reproject fim_geoid to EPSG:5070, NAD83 / Conus Albers (meters)
    fim_geoid_gdf = fim_geoid_gdf.to_crs(epsg=5070)
       
    # Calculate Bivariate Moran's I & Local Moran's I
    print(f"{dam_id}: Step 2, 2/2, Calculating Moran\'s I and LISA")
    mi_gdf, lm_gdf = calculate_bivariate_Moran_I_and_LISA(dam_id, census_dic, fim_geoid_gdf, fd_gdf)

    return mi_gdf, lm_gdf


def population_vulnerable_to_fim(dam_id, input_dir, fd_gdf, tract_gdf, census_dic, census_df):
    # Step 1: Compute GEOID of inundated and non-inundated regions from NID inundation mapping of each dam
    fim_geoid_gdf, fim_gdf = extract_fim_geoid(dam_id, input_dir, tract_gdf)

    # Step 2: Spatial correlation between fim and census data
    mi_gdf, lm_gdf = spatial_correlation(dam_id, fd_gdf, fim_geoid_gdf, census_dic, census_df)

    return fim_gdf, mi_gdf, lm_gdf


def population_vulnerable_to_fim_unpacker(args):
    return population_vulnerable_to_fim(*args)


##### ------------ Main Code Starts Here ------------ #####

if __name__ == "__main__":

    # How many dams will be run for each sbatch submission
    count_1 = 24
    count_2 = 8
    iter_num = int(sys.argv[1])
    if iter_num < 13:
        dam_count = count_1
        start_num = iter_num * count_1
        end_num = (iter_num + 1) * count_1
    else:
        dam_count = count_2
        start_num = 13 * count_1 + (iter_num-13) * count_2
        end_num = 13 * count_1 + (iter_num-13 + 1) * count_2

    print(f"Iter: {iter_num}, Dam count: {dam_count}, Range: {start_num} - {end_num}")
    PROCESSORS = dam_count    

    # Multiple Scenarios
    sce_mh = {'loadCondition': 'MH', 'breachCondition': 'F'}  # Maximun Height scenario
    sce_tas = {'loadCondition': 'TAS', 'breachCondition': 'F'}  # Top of Active Storage scenario
    sce_nh = {'loadCondition': 'NH', 'breachCondition': 'F'}  # Normal Height scenario

    # Local directory
    # data_dir = os.getcwd()
    # output_dir = os.path.join(data_dir, f'Multi_F_Results', f'N_{iter_num}')
    
    # Anvil directory
    data_dir = '/anvil/projects/x-cis220065/x-cybergis/compute/Aging_Dams'
    output_dir = os.path.join(data_dir, f'Multi_F_Results_NWFim_Queen', f'N_{iter_num}')
    print('Output Directory: ', output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # API_Key = '' # Census api key

    # Find the list of dams in the input folder
    fed_dams = pd.read_csv('./nid_available_scenario.csv')

    # Remove dams with error (fim is too small to generate)
    fed_dams = fed_dams.loc[fed_dams['ID'] != 'CO01283S001']

    # Select only Fed Dams that have inundation maps.
    for sce in [sce_mh, sce_tas, sce_nh]:
        fed_dams = fed_dams.loc[fed_dams[f'{sce["loadCondition"]}_{sce["breachCondition"]}'] == True]
        fed_dams = fed_dams.loc[fed_dams.apply(lambda x: True 
                                            if os.path.exists(os.path.join(data_dir, 
                                            f'NID_FIM_{sce["loadCondition"]}_{sce["breachCondition"]}', 
                                            f'{sce["loadCondition"]}_{sce["breachCondition"]}_{x["ID"]}.tiff')
                                            ) else False, axis=1)].reset_index(drop=True)
    fed_dams = gpd.GeoDataFrame(fed_dams, geometry=gpd.points_from_xy(fed_dams['LON'], fed_dams['LAT'], crs="EPSG:4326"))
    print(f'Total Dams: {fed_dams.shape[0]}')
    
    dois = fed_dams['ID'].to_list()
    dois = dois[start_num:end_num]
    print(dois)

    # Census tract to find state associated with fim of each dam
    tract = gpd.read_file(os.path.join(data_dir, 'census_geometry', 'census_tract_from_api.geojson'))

    # List of census data to be retrieved. The key is the census data abbreviation used in the final table.
    # str: single variable
    # list: [[To be summed and set as numerator], demonimator]  
    census_info = {
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

    # Retrieved census data (colume names are equal to census_info.keys())
    census_data = pd.read_csv(os.path.join(data_dir, 'census_geometry', 'census_data.csv'))
    census_data['GEOID_T'] = census_data['GEOID_T'].apply(lambda x:"{:011d}".format(x))

    # Empty GeoDataFrame for storing the results
    fim_output = pd.DataFrame() # GEOID of inundated and non-inundated regions 
    mi_result = pd.DataFrame() # Bivariate Moran's I result
    lm_result = pd.DataFrame() # Bivariate LISA result

    pool = mp.Pool(PROCESSORS)

    '''
    ## Results overview
    fim_gdf: GEOID of inundated and non-inundated regions
    mi_gdf: Bivariate Moran's I result
    lm_gdf: Bivariate LISA result
    '''
    results = pool.map(population_vulnerable_to_fim_unpacker,
                            zip(dois, # List of Dam_ID                                
                                itertools.repeat(data_dir), # Input directory of NID inundation mapping
                                itertools.repeat(fed_dams), # GeoDataFrame of all dams
                                itertools.repeat(tract), # GeoDataFrame of census tracts
                                itertools.repeat(census_info), # Dictionary of census data to be retrieved
                                itertools.repeat(census_data) # Census API key
                                )
                       )

    pool.close()

    print(f'Merging results for {len(results)} dams')

    for result in results:
        fim_output = pd.concat([fim_output, result[0]]).reset_index(drop=True)
        mi_result = pd.concat([mi_result, result[1]]).reset_index(drop=True)
        lm_result = pd.concat([lm_result, result[2]]).reset_index(drop=True)

    lm_result = lm_result.to_crs(epsg=4326)

    fim_output.to_file(os.path.join(output_dir, f"Multi_F_fim.geojson"), driver='GeoJSON')
    mi_result.to_file(os.path.join(output_dir, f"Multi_F_mi.geojson"), driver='GeoJSON')
    lm_result.to_file(os.path.join(output_dir, f"Multi_F_lm.geojson"), driver='GeoJSON')

    print('Done')
