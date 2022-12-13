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




class ExtractInundationCensusTracts(): # GeoEDFPlugin):
    __optional_params = ['load_condition','breach_condition']
    __required_params = ['dam_ids','floodmap_path', 'target_path']
    
    __census_filepath = './census_geometry' # '/compute_shared/Aging_Dams/census_geometry/'
    __API_Key = 'fbcac1c2cc26d853b42c4674adf905e742d1cb2b'
    

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params
        print(self.provided_params)
        print(kwargs)
        
        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise AttributeError('Required parameter %s for ExtractInundationCensusTracts not provided' % param)
                # raise GeoEDFError('Required parameter %s for ExtractInundationCensusTracts not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))
            
        # initialize some other DFs
        fed_dams_data = requests.get('https://fim.sec.usace.army.mil/ci/fim/getAllEAPStructure').json()
        self.fed_dams = pd.DataFrame(fed_dams_data)
        self.fed_dams = gpd.GeoDataFrame(self.fed_dams, geometry=gpd.points_from_xy(self.fed_dams['LON'], self.fed_dams['LAT'], crs="EPSG:4326"))

        # class super class init
        super().__init__()
        
    # find the scenario ID given a load, breach condition and dam ID
    def find_scenario_id(self,load_condition,breach_condition,dam_id):
        r = requests.get("https://fim.sec.usace.army.mil/ci/fim/getEAPLayers?id=" + dam_id)
    
        if r.status_code == 200:
            scenarios = json.loads(r.content)
            for scene_num in range(len(scenarios)):
                loadCondition = scenarios[scene_num]['loadCondition']
                breachCondition = scenarios[scene_num]['breachCondition']
                if (loadCondition == load_condition) and (breachCondition == breach_condition):
                    return scene_num
            return np.nan
        else:
            return np.nan        


    def resample_raster(self, rasterfile, filename, rescale_factor):
        # first determine pixel size to resample to 10x
        xres = 0
        yres = 0
        try:
            out = subprocess.run(["gdalinfo","-json",rasterfile],stdout=subprocess.PIPE)
            raster_meta = json.loads(out.stdout.decode('utf-8'))
            if 'geoTransform' in raster_meta:
                xres = raster_meta['geoTransform'][1]
                yres = raster_meta['geoTransform'][5]
                xres = xres * rescale_factor
                yres = yres * rescale_factor
            else:
                raise AttributeError('Error determining pixel size for raster file')
                # raise GeoEDFError('Error determining pixel size for raster file')
        except:
            raise AttributeError('Error determining pixel size for raster file')
            # raise GeoEDFError('Error determining pixel size for raster file')
            
        if (xres != 0) and (yres != 0):
            # resample raster
            save_path = self.target_path +"/"+ filename + "_resample.tiff"
            subprocess.run(["gdalwarp","-r","bilinear","-of","GTiff","-tr",str(xres),str(yres),rasterfile,save_path])

            return save_path
        else:
            raise AttributeError('Resampled raster does not have valid pixel resolution')
            # raise GeoEDFError('Resampled raster does not have valid pixel resolution')


    # reclassify, resample, and polygonize raster flood inundation map
    def polygonize_fim(self,rasterfile):
            
        filename = rasterfile.split("/")[-1].split(".")[-2]
            
        # resample the raster file
        resample_10_path = self.resample_raster(rasterfile, filename, rescale_factor=10)
            
        # now reclassify raster
        water_lvl = [0, 2, 6, 15, np.inf]  # Original inundation map value (underwater in feet)
        water_lvl_recls = [-9999, 1, 2, 3, 4]
        reclass_file = self.target_path + "/" + filename + "_reclass.tiff"
        outfile = "--outfile="+reclass_file
        subprocess.run(["gdal_calc.py","-A",resample_10_path,outfile,"--calc=-9999*(A<=0)+1*((A>0)*(A<=2))+2*((A>2)*(A<=6))+3*((A>6)*(A<=15))+4*(A>15)","--NoDataValue=-9999"],stdout=subprocess.PIPE)
            
        # now polygonize the reclassified raster
        geojson_out = "%s/%s.json" % (self.target_path,filename)
        subprocess.run(["gdal_polygonize.py",reclass_file,"-b","1", "-f", "GeoJSON", geojson_out, filename,"value"])
             
        inundation_polygons = gpd.read_file(geojson_out)
            
        inundation_polygons = inundation_polygons.loc[inundation_polygons['value'] != -9999]  # Remove pixels of null value

        # drop invalid geometries
        inundation_polygons = inundation_polygons.loc[inundation_polygons['geometry'].is_valid, :]
            
        # Entire coverage of inundation map
        inundation_dis_geom = inundation_polygons.geometry.unary_union

        # Coverage for each class of inundation map
        inundation_per_cls = inundation_polygons.dissolve(by='value')
        inundation_per_cls.reset_index(inplace=True)

        # Save the polygonized results
        #poly_out = "%s/%s.geojson" % (self.target_path,filename)
        #inundation_polygons.to_file(poly_out)
            
        # remove all temp files
        os.remove(resample_10_path)
        os.remove(reclass_file)
        os.remove(geojson_out)
            
        return inundation_per_cls
            

    def calculate_ellipse_based_on_convex_hull(self, points_ary):

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


    def fim_and_ellipse(self, dam_id, scene):
        
        fim_path = f"{self.floodmap_path}/{scene['loadCondition']}_{scene['breachCondition']}_{dam_id}.tiff"

        fim_gdf = self.polygonize_fim(fim_path)
        fim_gdf['Dam_ID'] = dam_id
        fim_gdf['Scenario'] = f"{scene['loadCondition']}_{scene['breachCondition']}"

        # Collecting points from convex hull of the inundation map
        # These points will be used for calculating mvee 
        convex_hull_pnts = np.array(fim_gdf.unary_union.convex_hull.exterior.coords)
        ellipse = self.calculate_ellipse_based_on_convex_hull(convex_hull_pnts)
        ellipse_gdf = gpd.GeoDataFrame({'Dam_ID':f'{dam_id}'}, index=[0], geometry=[ellipse], crs='EPSG:4326')

        return fim_gdf, ellipse_gdf


    def state_num_related_to_fim(self, ellipse_gdf, tract_gdf):

        tract_geoms = pygeos.from_shapely(tract_gdf['geometry'].values)
        tract_geoms_tree = pygeos.STRtree(tract_geoms, leafsize=50)
        
        ellipse_geom = pygeos.from_shapely(ellipse_gdf['geometry'].values[0])    
        query_intersect = tract_geoms_tree.query(ellipse_geom, predicate='intersects')
        tract_gdf = tract_gdf.loc[query_intersect]

        tract_gdf['STATE'] = tract_gdf.apply(lambda x:x['GEOID'][0:2], axis=1)
        unique_state = tract_gdf['STATE'].unique()

        # return type: list
        return unique_state


    def extract_fim_geoid(self, dam_id, scene, tract_gdf):
        print(f'{dam_id}: Step 1, 1/4, Identifying associated regions (Ellipse)')
        fim_gdf, ellipse_gdf = self.fim_and_ellipse(dam_id, scene)

        print(f'{dam_id}: Step 1, 2/4, Search states associated')
        fim_state = self.state_num_related_to_fim(ellipse_gdf, tract_gdf)
        print(f'-- {dam_id} impacts {len(fim_state)} States, {fim_state}')

        if len(fim_state) == 1: # If only one state is associated with the inundation mapping
            census_gdf = gpd.read_file(os.path.join(self.__census_filepath,f'tl_2020_{fim_state[0]}_tabblock20.geojson'))
        elif len(fim_state) >= 2: # If multiple states are associated with the inundation mapping
            census_gdf = pd.DataFrame()
            for state_num in fim_state:
                temp_gdf = gpd.read_file(os.path.join(self.__census_filepath,f'tl_2020_{state_num}_tabblock20.geojson'))
                census_gdf = pd.concat([temp_gdf, census_gdf]).reset_index(drop=True)
                census_gdf = gpd.GeoDataFrame(census_gdf, geometry=census_gdf['geometry'], crs="EPSG:4326")
        else:
            raise AttributeError('NO STATE is related to Inundation Mapping')

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
                                        group_keys=False).apply(lambda x:x.loc[x['Class'].idxmax()]).reset_index(drop=True)
        fim_geoid_gdf = fim_geoid_gdf.set_crs(epsg=4326)

        return fim_geoid_gdf, fim_gdf, ellipse_gdf


    def extract_fim_geoid_unpakcer(self, args):
        return self.extract_fim_geoid(*args)


    def process(self):
        
        # set load and breach conditions if null
        if self.load_condition is None:
            self.load_condition = 'TAS'
        if self.breach_condition is None:
            self.breach_condition = 'F'
            
    
        # create a scene dict
        scene = {'loadCondition':self.load_condition,'breachCondition':self.breach_condition}
            
        # prepare results dfs
        fim_output = pd.DataFrame() # GEOID of inundated and non-inundated regions 
        ellipse_output = pd.DataFrame() # Ellipse of inundation mapping
        mi_result = pd.DataFrame() # Bivariate Moran's I result
        lm_result = pd.DataFrame() # Bivariate LISA result

        print("Prepare census data frame")
        
        # prepare census df
        tract = gpd.read_file(os.path.join(self.__census_filepath, 'census_tract_from_api.geojson'))
            
        print("Census data frame prepared, now run processes")
        
        # determine dam_ids to process
        print(f"Dam IDs to process: {self.dam_ids}")
        # dam_ids = self.dam_ids.split(',')
        dam_ids = self.dam_ids

        # try:
        pool = mp.Pool(1)
        results = pool.map(self.extract_fim_geoid_unpakcer,
                        zip(dam_ids, # List of Dam_ID
                            itertools.repeat(scene), # Dam failure scenario
                            itertools.repeat(tract) # GeoDataFrame of census tracts
                            )
                            )
        pool.close()
                
        # merge results
        for result in results:
            fim_output = pd.concat([fim_output, result[0]]).reset_index(drop=True)
            ellipse_output = pd.concat([ellipse_output, result[1]]).reset_index(drop=True)
            mi_result = pd.concat([mi_result, result[2]]).reset_index(drop=True)
            # lm_result = pd.concat([lm_result, result[3]]).reset_index(drop=True)
                    
        fim_output.to_file(os.path.join(self.target_path, f"{scene['loadCondition']}_{scene['breachCondition']}_fim.geojson"), driver='GeoJSON')
        ellipse_output.to_file(os.path.join(self.target_path, f"{scene['loadCondition']}_{scene['breachCondition']}_ellipse.geojson"), driver='GeoJSON')
        mi_result.to_file(os.path.join(self.target_path, f"{scene['loadCondition']}_{scene['breachCondition']}_mi.geojson"), driver='GeoJSON')
        # lm_result.to_file(os.path.join(self.target_path, f"{scene['loadCondition']}_{scene['breachCondition']}_lm.geojson"), driver='GeoJSON')

        # except:
        #     raise AttributeError("Error occurred when processing inundation maps for census tracts")
            # raise GeoEDFError("Error occurred when processing inundation maps for census tracts")
            

breach_condition = 'F'
load_condition = 'TAS'

processor_dict = {'ExtractInundationCensusTracts':
                              {'breach_condition': breach_condition,
                               'load_condition': load_condition,
                               'floodmap_path': './NID_FIM_'+load_condition+'_'+breach_condition,
                               'dam_ids': ['TX00009'],
                               'target_path': './NID_FIM_'+load_condition+'_'+breach_condition }
                               }

                
print(processor_dict['ExtractInundationCensusTracts'])


if __name__ == "__main__":
    test_JP = ExtractInundationCensusTracts(**processor_dict['ExtractInundationCensusTracts'])
    test_JP.process()