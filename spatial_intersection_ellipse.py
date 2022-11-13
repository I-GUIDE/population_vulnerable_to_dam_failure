import rioxarray
import qinfer
import numpy as np
import numpy.linalg as la
import geopandas as gpd
import shapely.affinity
import subprocess
from osgeo import gdal
import pygeos
import json
import os
import pandas as pd
import multiprocessing as mp
import requests
import itertools


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


# reclassify, resample, and polygonize raster flood inundation map
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
    

def extract_inundated_area_geoid(input_dir, census_gdf, dam_id, scene):
    
    fim_path = f"./{input_dir}/{scenarios['loadCondition']}_{scenarios['breachCondition']}_{dam_id}.tiff"

    # Destination dataframe to save the results
    fim_geoid_df = pd.DataFrame({'Dam_ID': pd.Series(dtype='str'),
                                'Scenario': pd.Series(dtype='str'),
                                'GEOID': pd.Series(dtype='str'),
                                'Class': pd.Series(dtype='str')}
                                )        

    print(f"{dam_id}: 1/4, Polygonizing inundation map")
    fim_gdf = polygonize_fim(fim_path)
    fim_gdf['Dam_ID'] = dam_id
    fim_gdf['Scenario'] = f"{scene['loadCondition']}_{scene['breachCondition']}"

    # Create STRtree for census_gdf
    print(f"{dam_id}: 2/4, Extracting inundated census blocks")
    census_geoms = pygeos.from_shapely(census_gdf['geometry'].values)
    census_geoms_tree = pygeos.STRtree(census_geoms, leafsize=50)

    # Extract census tract intersecting with each class of inundation map
    for water_cls in fim_gdf['value'].unique():
        inund_per_cls_geom = pygeos.from_shapely(fim_gdf.loc[fim_gdf['value'] == water_cls, 'geometry'].values[0])
        query_inund_census_geom = census_geoms_tree.query(inund_per_cls_geom, predicate='intersects')
        inund_census_gdf = census_gdf.loc[query_inund_census_geom]

        for geoid_ in inund_census_gdf['GEOID'].to_list():
            new_row = pd.DataFrame({'Dam_ID': dam_id, 
                                    'Scenario': f"{scene['loadCondition']}_{scene['breachCondition']}", 
                                    'GEOID': geoid_, 
                                    'Class': water_cls}, 
                                    index=[0]
                                    )
            fim_geoid_df = pd.concat([new_row, fim_geoid_df]).reset_index(drop=True)
    
    # Caclulate minimum-volume enclosing ellipse (mvee) of the inundation map to extract benchmark area
    print(f"{dam_id}: 3/4, Extracting benchmark area")

    # Collecting points from convex hull of the inundation map
    # These points will be used for calculating mvee 
    convex_hull_pnts = np.array(fim_gdf.unary_union.convex_hull.exterior.coords)
    ellipse = calculate_ellipse_based_on_convex_hull(convex_hull_pnts)
    ellipse_gdf = gpd.GeoDataFrame({'Dam_ID':f'{dam_id}'}, index=[0], geometry=[ellipse], crs='EPSG:4326')
    
    # Extract benchmark area (not inundated) intersecting with the ellipse
    ellipse_geom = pygeos.from_shapely(ellipse)    
    query_benchmark_census_geom = census_geoms_tree.query(ellipse_geom, predicate='intersects')
    benchmark_census_gdf = census_gdf.loc[query_benchmark_census_geom]

    for geoid_ in benchmark_census_gdf['GEOID'].to_list():
        new_row = pd.DataFrame({'Dam_ID': dam_id, 
                                'Scenario': f"{scene['loadCondition']}_{scene['breachCondition']}", 
                                'GEOID': geoid_, 
                                'Class': 0
                                }, index=[0]
                                )
        fim_geoid_df = pd.concat([new_row, fim_geoid_df]).reset_index(drop=True)

    print(f"{dam_id}: 4/4, Assigning geometry to census blocks")
    fim_geoid_gdf = fim_geoid_df.merge(census_gdf, on='GEOID')
    fim_geoid_gdf = gpd.GeoDataFrame(fim_geoid_gdf, geometry=fim_geoid_gdf['geometry'], crs='EPSG:4326')
    fim_geoid_gdf['Class'] = fim_geoid_gdf['Class'].astype(int)
    fim_geoid_gdf = fim_geoid_gdf.groupby(['Dam_ID', 'Scenario', 'GEOID'], 
                                    group_keys=False).apply(lambda x:x.loc[x['Class'].idxmax()]
                                                            ).reset_index(drop=True)

    return fim_geoid_gdf, fim_gdf, ellipse_gdf


def extract_inundated_area_geoid_unpacker(args):
    return extract_inundated_area_geoid(*args)



##### ------------ Main Code Starts Here ------------ #####

PROCESSORS = 2
cwd = os.getcwd()
input_path = 'NID_FIM_TAS_Breach'
scenarios = {'loadCondition': 'TAS', 'breachCondition': 'F'}
output_path = 'output'

# Load the census block 
census_gdf = gpd.read_file(os.path.join(cwd, 'census_geometry', 'tl_2020_block_texas.geojson'))
# census_gdf = gpd.read_file(os.path.join(cwd, 'census_geometry', 'tl_2020_tabblock20.geojson'))

# Clean census_gdf for having GEOID column and geometry in EPSG:4326
if 'GEOID' in census_gdf.columns:
    pass
elif 'geoid' in census_gdf.columns:
    census_gdf.rename(columns={'geoid': 'GEOID'}, inplace=True) # Comment this line for `tl_2020_tabblock20.geojson`
else:
    raise AttributeError('either GEOID or geoid column is necessary')

if census_gdf.crs != 'EPSG:4326':
    census_gdf = census_gdf.to_crs(epsg=4326)

census_gdf = census_gdf[['GEOID', 'geometry']]

# Find the list of dams in the input folder
fed_dams = requests.get('https://fim.sec.usace.army.mil/ci/fim/getAllEAPStructure').json()
fd_df = pd.DataFrame(fed_dams)
dois = fd_df['ID'].to_list()
print(f"Total of {len(dois)} dams")

dois = [doi for doi in dois if os.path.exists(os.path.join(cwd, input_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_{doi}.tiff"))]
dois = ['TX00004', 'TX00006']
print(f"Dam of Interest counts: {len(dois)}")
print(dois)

# Empty GeoDataFrame for storing the results
fim_geoid_gdf_output = gpd.GeoDataFrame()
fim_gdf_output = gpd.GeoDataFrame()
ellipse_gdf_output = gpd.GeoDataFrame()

if __name__ == "__main__":
    pool = mp.Pool(PROCESSORS)
    results = pool.map(extract_inundated_area_geoid_unpacker,
                            zip(itertools.repeat(input_path),
                                itertools.repeat(census_gdf),
                                dois, # list of DAM ID 
                                itertools.repeat(scenarios))
                       )

    pool.close()

    print(f'Merging results for {len(results)} dams')

    for result in results:
        fim_geoid_gdf_output = pd.concat([fim_geoid_gdf_output, result[0]]).reset_index(drop=True)
        fim_gdf_output = pd.concat([fim_gdf_output, result[1]]).reset_index(drop=True)
        ellipse_gdf_output = pd.concat([ellipse_gdf_output, result[2]]).reset_index(drop=True)

    fim_geoid_gdf_output.to_file(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_fim_geoid.geojson"), driver='GeoJSON')
    fim_gdf_output.to_file(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_fim.geojson"), driver='GeoJSON')
    ellipse_gdf_output.to_file(os.path.join(cwd, output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_ellipse.geojson"), driver='GeoJSON')

'''
# Testing code for a dam
dam_id = 'TX00006'
fim_geoid_gdf, fim_gdf, ellipse = extract_inundated_area_geoid(input_path, census_gdf, dam_id, scenarios)

fim_geoid_gdf.to_file(os.path.join(output_path, f"fim_geoid_{scenarios['loadCondition']}_{scenarios['breachCondition']}_{dam_id}.geojson"), driver='GeoJSON')
fim_gdf.to_file(os.path.join(output_path, f"fim_{scenarios['loadCondition']}_{scenarios['breachCondition']}_{dam_id}.geojson"), driver='GeoJSON')
ellipse.to_file(os.path.join(output_path, f"{scenarios['loadCondition']}_{scenarios['breachCondition']}_{dam_id}_ellipse.geojson"), driver='GeoJSON')

'''