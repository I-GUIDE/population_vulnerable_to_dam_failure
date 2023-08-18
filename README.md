# Who is facing the potential risk of aging dam failure? Focusing on 345 dams in the conterminous United States.

Authors: Jinwoo Park <sup>1,2</sup>, Shaowen Wang <sup>2,3,*</sup>, Courtney Flint <sup>4</sup>, and Upmanu Lall <sup>5</sup> <br>
<sup>1</sup> Department of Geography and Geographic Information Science, University of North Dakota <br>
<sup>2</sup> CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois Urbana-Champaign <br>
<sup>3</sup> Department of Geography and Geographic Information Science, Illinois Urbana-Champaign <br>
<sup>4</sup> Department of Environment and Society, Utah State University <br>
<sup>5</sup> Department of Earth and Environmental Engineering, Columbia University <br>
Correspondence(<sup>\*</sup>): Shaowen Wang, shaowen@illinois.edu


Last Updated Date: August 18, 2023

## Abstract: 
The dam infrastructure in the conterminous United States (CONUS) has exceeded its designed service lives to a large extent, posing an increased risk of failures that can cause catastrophic disasters with substantial economic and human losses. However, limited attention has been paid to the characteristics of at-risk populations, hindering adequate understanding and preparedness for emergency planning. Our study proposes a framework employing spatial metrics to discover where and whether socially vulnerable populations are more exposed to flood inundation risks induced by dam failures. By applying the framework to 345 dams in the CONUS, we found that characteristics of at-risk populations vary extensively across space. To better understand this spatial variability, we categorized the dams into five clusters based on at-risk population characteristics. We find that of the dams analyzed, those in California, New England, and the Upper Mississippi basin, pose particularly high consequential risks for socially vulnerable populations.

## Keywords: 
Social vulnerability, Inundation risk, Aging infrastructure, Dam failures, Spatial metrics

## Featured Figure <br>
![](./images/cluster_geog.jpg)
Dam locations and the portion of clusters per hydrological unit region. <br>
Note: 345 dams displayed in the figure are a sample of entire dams in the CONUS (~92,000), as only those dams have inundation maps available on the NID. Point colors in the map indicate the clusters of dams. The pie charts demonstrate the percentage of clusters in each hydrological unit region, and the number in the center means dam counts in each region. 

- Out of 345 dams, each cluster has the percentage as follows: Cluster A (19%), Cluster B (13 %), Cluster C (12 %), Cluster D (16 %), and Cluster E (40 %). 
- A significant portion of dams in Upper Mississippi (50%), California (44%), New England (36%), and Texas-Gulf (32%) regions were classified as Cluster A. 
- A substantial percentage of Cluster B (i.e., dams potentially affecting advantaged populations) was observed in Missouri (36%) and Texas-Gulf (27%) regions, while its overall percentage was 13%. 
- Ohio and Arkansas-White-Red regions are the regions that have the largest and the second largest number of dams, but they provided a similar proportion of dams with the overall percentage. 

## Employed Data <br>
- `sample_data` folder contains sample data required for `Local_Analysis_Multi_Scenario.ipynb`.
- the original data is stored in Keeling server (/data/cigi/common/jparkgeo/aging_dam_data).
    - NID_FIM_MH_F: Inundation maps induced by dam failures under the Maximum Height (MH) and Breach (F) scenario. <br>
    Source: National Inventory of Dams (NID) (https://nid.sec.usace.army.mil/viewer/index.html).
    - NID_FIM_TAS_F: Inundation maps induced by dam failures under the Normal Height (NH) and Breach (F) scenario.  <br>
    Source: National Inventory of Dams (NID) (https://nid.sec.usace.army.mil/viewer/index.html).
    - NID_FIM_NH_F: Inundation maps induced by dam failures under the Top of Active Storage (TAS) and Breach scenario. <br>
    Source: National Inventory of Dams (NID) (https://nid.sec.usace.army.mil/viewer/index.html).
    - census_geometry: Geometry of census block, census tract, and state. <br>
    Source: US Census Bureau (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)