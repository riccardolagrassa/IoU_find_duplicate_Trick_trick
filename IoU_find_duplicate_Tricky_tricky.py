'''
IoU_find_duplicate_Tricky_tricky
Copyright (C) 26/07/23 Riccardo La Grassa

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import multiprocessing
import time
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.ops import  unary_union
from multiprocessing import Pool
import geopandas as gpd


def ReadYolo_CraterCSV(filename, confidence):
    """
        ReadYolo_CraterCSV(filename, confidence)

        Read and process a CSV file containing information about craters identified by YOLOv8.

        Parameters:
            filename (str): The path to the CSV file containing crater information.
            confidence (float): The minimum confidence level for crater identification.

        Returns:
            pandas.DataFrame: A DataFrame containing filtered and processed crater data.
        """
    craters = pd.read_csv(filename, header=0, usecols=list(range(0, 7)))
    #polygons_ortho = [wkt.loads(geom_str) for geom_str in craters['Shapefile_ortho']]
    polygons_cyl = [wkt.loads(geom_str) for geom_str in craters['Shapefile_cyl']]
    #craters['Shapefile_ortho'] = polygons_ortho
    craters['Shapefile_cyl'] = polygons_cyl
    craters = craters[(craters['Confidence'].str.replace('[', '').str.replace(']', '').str.strip().astype(float)) >= confidence]
    craters.sort_values(by='Area', inplace=True, ascending=False)
    craters.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    print("YoloV8 craters loaded: ", len(craters))
    return craters


def find_matching_elements_parallel(args):
    """
        find_matching_elements_parallel(args)

        Parallel function to find matching elements (craters) in a DataFrame using Shapely spatial operations.

        Parameters:
            args (tuple): A tuple containing DataFrame chunk, row_duplicate list, and Shapefile_cyl list.

        Returns:
            pandas.DataFrame: A DataFrame containing matching elements with merged geometries.
        """
    df1, row_duplicate, Shapefile_cyl_list = args
    matching_elements = pd.DataFrame(columns=('Index', 'Longitude', 'Latitude', 'Confidence', 'Shapefile_cyl', 'Match_row'))
    for row1 in df1.itertuples(index=True):
        row_duplicate = []
        Shapefile_cyl_list = []
        try:
            nearby_indexes = list(spatial_index.intersection(row1.Shapefile_cyl.centroid.buffer(0.001).bounds))
        except:
            continue
        df2_window = craters_full_load.iloc[nearby_indexes]
        Shapefile_cyl_list.append(row1.Shapefile_cyl)
        for row2 in df2_window.itertuples(index=True):
            if row1.Shapefile_cyl.intersects(row2.Shapefile_cyl) and \
                    not row1.Shapefile_cyl.contains(row2.Shapefile_cyl) and \
                    not row2.Shapefile_cyl.contains(row1.Shapefile_cyl):
                iou = row1.Shapefile_cyl.intersection(row2.Shapefile_cyl).area / row1.Shapefile_cyl.union(row2.Shapefile_cyl).area
                if iou >= 0.6 and iou!=1.0:
                    row_duplicate.append(row2.Index)
        if len(row_duplicate)>0:
            all_geometries = df2_window.loc[row_duplicate, 'Shapefile_cyl']
            all_geometries.loc[all_geometries.index[-1] + 1] = Shapefile_cyl_list[0]
            merged_polygons = unary_union(all_geometries)
            merged_polygons = merged_polygons.buffer(-abs(0.01))
        else:
            merged_polygons = row1.Shapefile_cyl.buffer(-abs(0.01))
            row_duplicate = []

        matching_elements = matching_elements.append({
            'Index': row1.Index,
            'Longitude': row1.Longitude,
            'Latitude': row1.Latitude,
            'Confidence': row1.Confidence,
            'Shapefile_cyl': merged_polygons,
            'Match_row': row_duplicate,
        }, ignore_index=True)

    return matching_elements


def parallelize_find_matching_elements(df1, num_processes):
    """
        parallelize_find_matching_elements(df1, num_processes)

        Parallelizes the process of finding matching elements (craters) in a DataFrame using multiprocessing.

        Parameters:
            df1 (pandas.DataFrame): The DataFrame containing crater information to be processed (chunks).
            num_processes (int): The number of processes to run in parallel.

        Returns:
            pandas.DataFrame: A DataFrame containing matching elements with merged geometries.
        """
    args = [(chunk_df1, [], []) for chunk_df1 in np.array_split(df1, num_processes)]
    with Pool(num_processes) as pool:
        results = pool.map(find_matching_elements_parallel, args)
        pool.close()  # Close the pool to prevent any more tasks from being submitted
        pool.join()
    matching_elements = pd.concat(results)
    return matching_elements


confidence = 0.15
YolovLensMoon_csv_path = "/home/super/rlagrassa/YOLOv8/LROC_GlobalMoon_Craters_SEG_FoV_1_2_4_Global_Catalog/_GlobalCratersList.csv"
craters_full_load = ReadYolo_CraterCSV(YolovLensMoon_csv_path, confidence)
num_processes =  multiprocessing.cpu_count()

chunk_size = 10 #To avoid ram saturation we split into chunks the full dataframe and then we split again based to number of processes considered.
craters_full_load = gpd.GeoDataFrame(craters_full_load, geometry='Shapefile_cyl')
craters_chunked_load = [chunk_df1 for chunk_df1 in np.array_split(craters_full_load, chunk_size)]
#SRTree is used to order the full dataframe and get much faster the intersection craters around a craters centroid selected (see function find_matching_elements_parallel)
spatial_index = craters_full_load.sindex
start = time.time()
new_dataframe_list = []
for idx_chunk, chunk in enumerate(craters_chunked_load):
    print("Process chunk: ", idx_chunk, "/", chunk_size)
    local_start = time.time()
    new_dataframe_list.append(parallelize_find_matching_elements(chunk, num_processes=num_processes)) #epsilon_lon=23, epsilon_lat=0.41
    print("End Chunck time: ", time.time() - local_start)

redundancy_dataframe = pd.concat(new_dataframe_list, ignore_index=True)


#################################################
#Each row of the dataframe contains information about its duplicate craters,
#However we cannot eliminate from 'Match_row' column all information considered because we'll eliminate duplicate and the original one both.
#To avoid this, using cycle for, we put to -1 (in sequentially order) all the first duplication we got (avoiding to eliminate itself again).

#(e.g., Crater A -> Duplication C, F
#       Craters C -> Duplication A, F

#In such scenario the Craters A will eliminate C, F because duplicate of itself, however Craters C will eliminate A and F (so "A" will never exists, having an error)
#The list of elimination is considered: [C, F, A, F], however since we proceeded sequentially we remove index as first as we can get obtaining: [C, F]
#Finally, Craters C cannot remove again Craters A in cycle redundancy maintaining the first one only.
#This approach is faster than considering a queue shared about different processing because of we do not wait the mutex variable to access to duplications world list.
#This trick allow us to obtain very fast results.
#################################################
print("Zeros row function started")
delete_indices = redundancy_dataframe['Match_row']
delete_indices = [item for sublist in delete_indices for item in sublist]
for i in delete_indices:
    delete_indices[i] = -1

delete_indices[:] = filter(lambda x: x != -1, delete_indices)
redundancy_dataframe.loc[delete_indices, :] = 0

print("Dropping all zeros row..")
final_dataframe = redundancy_dataframe.loc[~(redundancy_dataframe == 0).all(axis=1)] #Drop if all rows are 0 values
final_dataframe.to_csv('/home/super/rlagrassa/YOLOv8/Final_Filtered_Craters_SEG_FoV_1_2_4.csv', index=False)
print("Dropped Final Dataframe Len: ", len(final_dataframe))
#########################
# Convert latitude to meters
# lat_meter = latitude * 1737400
# lon_meter = longitude * moon_circumference / 360
final_dataframe = final_dataframe.drop('Match_row', axis=1)
gdf = gpd.GeoDataFrame(geometry=final_dataframe['Shapefile_cyl'])
gdf.to_file("/home/super/rlagrassa/YOLOv8/Final_LROC_FoV124_Filtered_shapeFile" + '_.shp', driver='ESRI Shapefile')
print("Time: ", time.time() - start)