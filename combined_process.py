import pandas as pd
import geopandas as gpd
import os
import requests
import esda
import libpysal
import multiprocessing as mp
import itertools
import qinfer
import shapely
import numpy as np
import pygeos
import numpy.linalg as la
import shapely.affinity
import subprocess
from osgeo import gdal
import json


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
        save_path = target_path +"/"+ filename + f"_resample_{rescale_factor}.tiff"
        subprocess.run(["gdalwarp","-r","bilinear","-of","GTiff","-tr",str(xres),str(yres),rasterfile_path,save_path])

        return save_path
    
def polygonize_fim(rasterfile_path):

    # Extract target path and filename from the given raster file path
    target_path = '/'.join(rasterfile_path.split('/')[:-1])
    filename = rasterfile_path.split("/")[-1].split(".")[-2]

    # Resample raster file to 10-times smaller
    resample_10_path = resample_raster(rasterfile_path, filename, target_path, rescale_factor=10)

    # Reclassify raster
    '''
    water_lvl = [0, 2, 6, 15, np.inf]  # Original inundation map value (underwater in feet)
    water_lvl_recls = [-9999, 1, 2, 3, 4]
    '''
    reclass_file = target_path + "/" + filename + "_reclass.tiff"
    outfile = "--outfile="+reclass_file
    subprocess.run(["gdal_calc.py","-A",resample_10_path,outfile,"--calc=-9999*(A<=0)+1*((A>0)*(A<=2))+2*((A>2)*(A<=6))+3*((A>6)*(A<=15))+4*(A>15)","--NoDataValue=-9999"],stdout=subprocess.PIPE)

    # Polygonize the reclassified raster
    geojson_out = "%s/%s.json" % (target_path, filename)
    subprocess.run(["gdal_polygonize.py", reclass_file, "-b", "1", geojson_out, filename, "value"])

    inund_polygons = gpd.read_file(geojson_out)
    inund_polygons = inund_polygons.loc[(inund_polygons['value'] != -9999) & (inund_polygons['value'] != 0)]  # Remove pixels of null value

    # drop invalid geometries
    inund_polygons = inund_polygons.loc[inund_polygons['geometry'].is_valid, :]

    # Coverage for each class of inundation map
    inund_per_cls = inund_polygons.dissolve(by='value')
    inund_per_cls.reset_index(inplace=True)

    # remove all temp files
    os.remove(resample_10_path)
    os.remove(reclass_file)
    os.remove(geojson_out)

    # inundation_per_cls: GeoDataFrame 
    return inund_per_cls


def calculate_ellipse_based_on_convex_hull(points_ary):
    
    # Calculate ellipse (MVEE; minimum-volume enclosing ellipse)
    A, centroid = qinfer.utils.mvee(points_ary)
    U, D, V = la.svd(A)
    
    ## x, y radii.
    rx, ry = 1./np.sqrt(D)

    ## Define major and minor semi-axis of the ellipse.
    dx, dy = 2 * rx, 2 * ry
    ma_axis, mi_axis = max(dx, dy), min(dx, dy)
    
    ## Calculate orientation of ellipse
    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))
    # Orientation angle (with respect to the x axis counterclockwise).
    alpha = arccos if arcsin > 0. else -1. * arccos

    ## Create a circle of radius 0.5 around center point:
    circ = shapely.geometry.Point(centroid).buffer(0.5)
    ellipse  = shapely.affinity.scale(circ, ma_axis, mi_axis)
    ellipse_rotate = shapely.affinity.rotate(ellipse, alpha)
    
    return ellipse_rotate


def fim_and_ellipse(dam_id, scene, input_dir):
        
    fim_path = f"./{input_dir}/{scenarios['loadCondition']}_{scenarios['breachCondition']}_{dam_id}.tiff"
    
    fim_gdf = polygonize_fim(fim_path)
    fim_gdf['Dam_ID'] = dam_id
    fim_gdf['Scenario'] = f"{scene['loadCondition']}_{scene['breachCondition']}"
    
    # Collecting points from convex hull of the inundation map
    # These points will be used for calculating mvee 
    convex_hull_pnts = np.array(fim_gdf.unary_union.convex_hull.exterior.coords)
    ellipse = calculate_ellipse_based_on_convex_hull(convex_hull_pnts)
    ellipse_gdf = gpd.GeoDataFrame({'Dam_ID':f'{dam_id}'}, index=[0], geometry=[ellipse], crs='EPSG:4326')
    
    return fim_gdf, ellipse_gdf


def state_num_related_to_fim(ellipse_gdf, tract_gdf):
    
    tract_geoms = pygeos.from_shapely(tract_gdf['geometry'].values)
    tract_geoms_tree = pygeos.STRtree(tract_geoms, leafsize=50)
       
    ellipse_geom = pygeos.from_shapely(ellipse_gdf['geometry'].values[0])    
    query_intersect = tract_geoms_tree.query(ellipse_geom, predicate='intersects')
    tract_gdf = tract_gdf.loc[query_intersect]
    
    tract_gdf['STATE'] = tract_gdf.apply(lambda x:x['GEOID'][0:2], axis=1)
    unique_state = tract_gdf['STATE'].unique()
    
    # return type: list
    return unique_state
    
    
def extract_fim_geoid(dam_id, scene, input_dir, tract_gdf):
    print(f'{dam_id}: Step 1, 1/4, Identifying associated regions (Ellipse)')
    fim_gdf, ellipse_gdf = fim_and_ellipse(dam_id, scene, input_dir)

    print(f'{dam_id}: Step 1, 2/4, Search states associated')
    fim_state = state_num_related_to_fim(ellipse_gdf, tract_gdf)
    print(f'-- {dam_id} impacts {len(fim_state)} States, {fim_state}')
    
    if len(fim_state) == 1: # If only one state is associated with the inundation mapping
        census_gdf = gpd.read_file(f'./census_geometry/tl_2020_{fim_state[0]}_tabblock20.geojson')
    elif len(fim_state) >= 2: # If multiple states are associated with the inundation mapping
        census_gdf = pd.DataFrame()
        for state_num in fim_state:
            temp_gdf = gpd.read_file(f'./census_geometry/tl_2020_{state_num}_tabblock20.geojson')
            census_gdf = pd.concat([temp_gdf, census_gdf]).reset_index(drop=True)
            census_gdf = gpd.GeoDataFrame(census_gdf, geometry=census_gdf['geometry'], crs="EPSG:4326")
    else:
        raise AttributeError('NO STATE is related to Inundation Mapping')

    # TODO: Remove line below
    census_gdf.rename(columns={'GEOID_B': 'GEOID'}, inplace=True)
    
    # Destination dataframe to save the results
    print(f"{dam_id}: Step 1, 3/4, Extracting GEOID of census blocks")
    fim_geoid_df = pd.DataFrame({'Dam_ID': pd.Series(dtype='str'),
                                'Scenario': pd.Series(dtype='str'),
                                'GEOID': pd.Series(dtype='str'),
                                'Class': pd.Series(dtype='str')}
                                )    
    
    # Create STRtree for census_gdf
    census_geoms = pygeos.from_shapely(census_gdf['geometry'].values)
    census_geoms_tree = pygeos.STRtree(census_geoms, leafsize=50)
    
    # Extract census tract intersecting with each class of inundation map
    for water_cls in fim_gdf['value'].unique():
        fim_geom_ = pygeos.from_shapely(fim_gdf.loc[fim_gdf['value'] == water_cls, 'geometry'].values[0])
        query_fim_geom_ = census_geoms_tree.query(fim_geom_, predicate='intersects')
        fim_geoid_ = census_gdf.loc[query_fim_geom_]

        for geoid_ in fim_geoid_['GEOID'].to_list():
            new_row = pd.DataFrame({'Dam_ID': dam_id, 
                                    'Scenario': f"{scene['loadCondition']}_{scene['breachCondition']}", 
                                    'GEOID': geoid_, 
                                    'Class': water_cls}, 
                                    index=[0]
                                    )
            fim_geoid_df = pd.concat([new_row, fim_geoid_df]).reset_index(drop=True)
            
    # Extract benchmark area (not inundated) intersecting with the ellipse
    ellipse_geom = pygeos.from_shapely(ellipse_gdf['geometry'].values[0])    
    query_non_fim_geom = census_geoms_tree.query(ellipse_geom, predicate='intersects')
    non_fim_geoid = census_gdf.loc[query_non_fim_geom]

    for geoid_ in non_fim_geoid['GEOID'].to_list():
        new_row = pd.DataFrame({'Dam_ID': dam_id, 
                                'Scenario': f"{scene['loadCondition']}_{scene['breachCondition']}", 
                                'GEOID': geoid_, 
                                'Class': 0
                                }, index=[0]
                                )
        fim_geoid_df = pd.concat([new_row, fim_geoid_df]).reset_index(drop=True) 
        
    print(f"{dam_id}: Step 1, 4/4, Assigning geometry to census blocks")
    fim_geoid_gdf = fim_geoid_df.merge(census_gdf, on='GEOID')
    fim_geoid_gdf = gpd.GeoDataFrame(fim_geoid_gdf, geometry=fim_geoid_gdf['geometry'], crs='EPSG:4326')
    fim_geoid_gdf['Class'] = fim_geoid_gdf['Class'].astype(int)
    fim_geoid_gdf = fim_geoid_gdf.groupby(['Dam_ID', 'Scenario', 'GEOID'], 
                                    group_keys=False).apply(lambda x:x.loc[x['Class'].idxmax()]
                                                            ).reset_index(drop=True)
    fim_geoid_gdf = fim_geoid_gdf.set_crs(epsg=4326)
    
    return fim_geoid_gdf, fim_gdf, ellipse_gdf


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

    return dam_local, fim_geoid_local


def spatial_correlation(dam_id, fd_gdf, fim_geoid_gdf, attr_dic, API_Key):
    
    # Retrieve census data from API
    print(f"{dam_id}: Step 2, 1/2, Retrieving census data")
    fim_geoid_gdf['GEOID_tract'] = fim_geoid_gdf.apply(lambda x:x['GEOID'][0:11], axis=1)
    for attr in attr_dic.keys():
        fim_geoid_gdf = census_data_of_fim_geoid(fim_geoid_gdf, attr, attr_dic, API_Key)

        # Replace not valid value (e.g., -666666) from census with nan value
        fim_geoid_gdf[attr] = fim_geoid_gdf.apply(lambda x: float('nan') if x[attr] < 0 else x[attr], axis=1)
        # print(f"-- Census Data ({attr}) is retrieved")

    # Reproject fim_geoid to EPSG:5070, NAD83 / Conus Albers (meters)
    fim_geoid_gdf = fim_geoid_gdf.to_crs(epsg=5070)
       
    # Calculate Bivariate Moran's I & Local Moran's I
    print(f"{dam_id}: Step 2, 2/2, Calculating Moran\'s I and LISA")
    mi_gdf, lm_gdf = calculate_bivariate_Moran_I_and_LISA(dam_id, attr_dic, fim_geoid_gdf, fd_gdf)

    return mi_gdf, lm_gdf


def population_vulnerable_to_fim(dam_id, scene, input_dir, fd_gdf, tract_gdf, attr_dic, API_Key):
    # Step 1: Compute GEOID of inundated and non-inundated regions from NID inundation mapping of each dam
    fim_geoid_gdf, fim_gdf, ellipse_gdf = extract_fim_geoid(dam_id, scene, input_dir, tract_gdf)

    # Step 2: Spatial correlation between fim and census data
    mi_gdf, lm_gdf = spatial_correlation(dam_id, fd_gdf, fim_geoid_gdf, attr_dic, API_Key)

    return fim_gdf, ellipse_gdf, mi_gdf, lm_gdf


def population_vulnerable_to_fim_unpacker(args):
    return population_vulnerable_to_fim(*args)




########## Main code starts here ##########


##### ------------ Main Code Starts Here ------------ #####

PROCESSORS = 4
cwd = os.getcwd()
input_dir = 'NID_FIM_TAS_Breach'
scenarios = {'loadCondition': 'TAS', 'breachCondition': 'F'}
output_dir = 'output'
API_Key = 'fbcac1c2cc26d853b42c4674adf905e742d1cb2b' # Census api key


# Find the list of dams in the input folder
fed_dams = requests.get('https://fim.sec.usace.army.mil/ci/fim/getAllEAPStructure').json()
fed_dams = pd.DataFrame(fed_dams)
fed_dams = gpd.GeoDataFrame(fed_dams, geometry=gpd.points_from_xy(fed_dams['LON'], fed_dams['LAT'], crs="EPSG:4326"))
dois = fed_dams['ID'].to_list()
# TODO: Uncomment the following line to run the code for all dams
dois = [doi for doi in dois if os.path.exists(os.path.join(cwd, input_dir, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_{doi}.tiff"))]

import random

dois = random.choices(dois, k=12)
print(f"Dam of Interest counts: {len(dois)}")
print(dois)


# Census tract to find state associated with fim of each dam
tract = gpd.read_file(os.path.join(cwd, 'census_geometry', 'census_tract_from_api.geojson'))

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


if __name__ == "__main__":

    # Empty GeoDataFrame for storing the results
    fim_output = pd.DataFrame() # GEOID of inundated and non-inundated regions 
    ellipse_output = pd.DataFrame() # Ellipse of inundation mapping
    mi_result = pd.DataFrame() # Bivariate Moran's I result
    lm_result = pd.DataFrame() # Bivariate LISA result


    pool = mp.Pool(PROCESSORS)

    '''
    ## Results overview
    fim_gdf: GEOID of inundated and non-inundated regions
    ellipse_gdf: Ellipse of inundation mapping
    mi_gdf: Bivariate Moran's I result
    lm_gdf: Bivariate LISA result
    '''
    results = pool.map(population_vulnerable_to_fim_unpacker,
                            zip(dois, # List of Dam_ID
                                itertools.repeat(scenarios), # Dam failure scenario
                                itertools.repeat(input_dir), # Input directory of NID inundation mapping
                                itertools.repeat(fed_dams), # GeoDataFrame of all dams
                                itertools.repeat(tract), # GeoDataFrame of census tracts
                                itertools.repeat(census_attr_dic), # Dictionary of census data to be retrieved
                                itertools.repeat(API_Key) # Census API key
                                )
                       )

    pool.close()

    print(f'Merging results for {len(results)} dams')

    for result in results:
        fim_output = pd.concat([fim_output, result[0]]).reset_index(drop=True)
        ellipse_output = pd.concat([ellipse_output, result[1]]).reset_index(drop=True)
        mi_result = pd.concat([mi_result, result[2]]).reset_index(drop=True)
        lm_result = pd.concat([lm_result, result[3]]).reset_index(drop=True)

    lm_result = lm_result.to_crs(epsg=4326)

    fim_output.to_file(os.path.join(cwd, output_dir, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_fim.geojson"), driver='GeoJSON')
    ellipse_output.to_file(os.path.join(cwd, output_dir, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_ellipse.geojson"), driver='GeoJSON')
    mi_result.to_file(os.path.join(cwd, output_dir, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_mi.geojson"), driver='GeoJSON')
    lm_result.to_file(os.path.join(cwd, output_dir, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_lm.geojson"), driver='GeoJSON')

    print('Done')
