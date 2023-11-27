import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import shapely
import geopandas as gpd

from .plot_utils import make_figax

def point_to_xy(point):
    return np.array(point.xy).reshape(-1)

class Neighborhoods():
    """
    Neighborhoods from geojson wrapper
    ---
    """

    def __init__(self):
        self.neighborhood_names = []
        self.neighborhoods = []
        self.neighborhood_samples = []

    def load_geojson(self, file_path: os.PathLike):
        """
        Load Geojson content to object
        ---
        Args:
        - file_path: Path to the geojson file
        """

        # Load file contents
        with open(file_path) as file:
            file_content = json.load(file)
        
        # Loop over neighborhoods in file
        for neighborhood in file_content["features"]:
            # Extract name
            neighborhood_name = neighborhood["properties"]["name"]

            # Load geometry
            neighborhood_geometry = neighborhood["geometry"]["coordinates"]
            # Support polygon & multipolygon
            neighborhood_geometrytype = neighborhood["geometry"]["type"]
            assert neighborhood_geometrytype in ["MultiPolygon", "Polygon"], f"{neighborhood_geometrytype = }"
            if neighborhood_geometrytype == "Polygon":
                neighborhood_geometry = [neighborhood_geometry]
            polygon_list = [None] * len(neighborhood_geometry)
            for i, coords in enumerate(neighborhood_geometry):
                # Generate polygon
                polygon_list[i] = shapely.polygons(coords[0])
            polygons = shapely.multipolygons(polygon_list)
            neighborhood_gdf = gpd.GeoDataFrame(index=["geometry"], geometry=[polygons])

            # Add neighborhood to object
            self.neighborhood_names.append(neighborhood_name)
            self.neighborhoods.append(neighborhood_gdf)
    
    def get_global_bounds(self):
        # Initialize
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        # Update with neighborhood bounds
        for neighborhood in self.neighborhoods:
            nx_min, ny_min, nx_max, ny_max = neighborhood.loc["geometry"].iloc[0].bounds
            x_min = min(x_min, nx_min)
            x_max = max(x_max, nx_max)
            y_min = min(y_min, ny_min)
            y_max = max(y_max, ny_max)
        
        return x_min, y_min, x_max, y_max

    def generate_samples(self, n_points=10000):
        # Reset samples
        self.neighborhood_samples = [None] * len(self.neighborhoods)

        # Generate random samples within global bounds of our neighborhoods
        x_min, y_min, x_max, y_max = self.get_global_bounds()
        x = np.random.uniform(x_min, x_max, n_points)
        y = np.random.uniform(y_min, y_max, n_points)

        # Intersect the points with each of the neighborhoods
        for i, neighborhood in enumerate(self.neighborhoods):
            # Turn points into geopandas object
            df = pd.DataFrame()
            df["points"] = list(zip(x, y))
            df["points"] = df["points"].apply(shapely.Point)
            points_gdf = gpd.GeoDataFrame(df, geometry="points")

            # Intersect with neighborhood
            Sjoin = gpd.tools.sjoin(points_gdf, neighborhood, predicate="within", how="left")
            intersection = points_gdf[Sjoin.index_right=="geometry"]
            points_in_neighborhood = np.array(list(intersection["points"].apply(point_to_xy)))
            self.neighborhood_samples[i] = points_in_neighborhood
        
        return self.neighborhood_samples


    def plot(self, ax=None, plot_samples = True):
        if ax is None:
            fig, ax = make_figax()
        
        for i, neighborhood in enumerate(self.neighborhoods):
            neighborhood_name = self.neighborhood_names[i]
            ax = neighborhood.boundary.plot(ax=ax, linewidth=1, label=neighborhood_name, color=f"C{i}")
            if self.neighborhood_samples and len(self.neighborhood_samples):
                samples = self.neighborhood_samples[i]
                if len(samples):
                    ax.scatter(samples[:, 0], samples[:, 1], c=f"C{i}", marker=".")
                else:
                    print(f"Warning: {neighborhood_name} has no samples")
        
        ax.set_title("Neighborhoods")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return ax
