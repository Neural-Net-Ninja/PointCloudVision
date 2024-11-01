import geopandas as gpd

def inspect_shapefile(file_path):
    gdf = gpd.read_file(file_path)
    print("Summary of the Shapefile:")
    print("-------------------------")
    print("First few rows of the data:")
    print(gdf.head())
    print()
    print("Attributes (Columns):")
    print(gdf.columns)
    print()
    print("Coordinate Reference System (CRS):")
    print(gdf.crs)
    print()
    print("Total number of features:")
    print(len(gdf))
    print()
    print("Geometry type:")
    print(gdf.geom_type.unique())
    print()
    print("Bounding box:")
    print(gdf.total_bounds)
    print()
    print("Summary of each attribute:")
    print(gdf.describe(include='all'))
    print()

if __name__ == "__main__":
    file_path = r'D:\Vector_analysis_Twin4road\shape_evaluation\extracted\ptv3_run5\arrows\Run 4S2224532_20220905_074227_0004_preprocessed_part0_pfeile.shp'
    inspect_shapefile(file_path)