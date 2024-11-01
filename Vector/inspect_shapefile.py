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
    file_path = 'path_to_your_shapefile.shp'
    inspect_shapefile(file_path)