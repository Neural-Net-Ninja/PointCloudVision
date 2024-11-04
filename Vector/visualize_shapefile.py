import geopandas as gpd
import matplotlib.pyplot as plt

def visualize_shapefile(file_path):
    gdf = gpd.read_file(file_path)
    
    # Plot the shapefile
    gdf.plot()
    plt.title("Visualization of the Shapefile")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

if __name__ == "__main__":
    file_path =  r'D:\Vector_analysis_Twin4road\shape_evaluation\gt\Run4_L_0_FN_Pfeile_1.shp'
    visualize_shapefile(file_path)