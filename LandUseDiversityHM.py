# Importing necessary libraries
import osmnx as ox  # Importing osmnx library for handling OpenStreetMap data
import geopandas as gpd  # Importing geopandas for working with geospatial data
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np  # Importing numpy for numerical computations
from shapely.geometry import Polygon  # Importing Polygon from shapely.geometry for geometry operations
import rasterio  # Importing rasterio for working with raster data
from rasterio.plot import show  # Importing show function from rasterio.plot for raster plotting

# Define a class for urban analysis
class LandUseDiversityHM:
    # Constructor method to initialize the class
    def __init__(self, north, south, east, west, num_tiles):
        # Initialize boundaries of the study area
        self.north = north  # Northern boundary of the study area
        self.south = south  # Southern boundary of the study area
        self.east = east  # Eastern boundary of the study area
        self.west = west  # Western boundary of the study area
        # Convert boundaries to a polygon
        self.polygon = ox.utils_geo.bbox_to_poly(bbox=(north, south, east, west))  # Converting boundaries to a polygon using osmnx
        # Initialize GeoDataFrames for buildings and land use
        self.gdf_buildings = None  # GeoDataFrame for buildings
        self.gdf_land_use = None  # GeoDataFrame for land use
        # Define number of tiles and calculate tile dimensions
        self.num_tiles = num_tiles  # Number of tiles covering the study area
        self.tile_width = (east - west) / num_tiles  # Width of each tile
        self.tile_height = (north - south) / num_tiles  # Height of each tile
        # Generate tiles covering the study area
        self.tiles = self.generate_tiles()  # Generating tiles

    # Method to acquire data including buildings, land use, and Landsat imagery
    def acquire_data(self, landsat_file=None):
        # Extract building footprints and land use polygons within the study area
        self.gdf_buildings = gpd.GeoDataFrame(
            ox.features_from_polygon(self.polygon, tags={'building': True})[['geometry', 'building']].query(
                'geometry.type == "Polygon"'))  # Extracting building footprints within the study area using osmnx

        self.gdf_land_use = gpd.GeoDataFrame((
            ox.features_from_polygon(self.polygon, tags={'landuse': True})[['geometry', 'landuse']]).query(
                'geometry.type == "Polygon"'))  # Extracting land use polygons within the study area using osmnx

        # Overlay tiles with buildings and land use to get their intersection
        buildings_in_tiles = gpd.overlay(self.tiles, self.gdf_buildings, how='intersection',keep_geom_type=False)  # Overlaying tiles with buildings
        land_use_in_tiles = gpd.overlay(self.tiles, self.gdf_land_use, how='intersection',keep_geom_type=False)  # Overlaying tiles with land use

        # Reproject GeoDataFrames from 4326 to Web Mercator (3857)
        self.tiles = self.tiles.to_crs(epsg=3857)  # Reprojecting tiles to Web Mercator
        self.gdf_buildings = buildings_in_tiles.to_crs(epsg=3857)  # Reprojecting buildings to Web Mercator
        self.gdf_land_use = land_use_in_tiles.to_crs(epsg=3857)  # Reprojecting land use to Web Mercator

        # If Landsat raster file is provided, open it
        if landsat_file:
            landsat_data = rasterio.open(landsat_file)  # Opening Landsat raster data
            self.landsat_data = landsat_data  # Storing Landsat raster data
        else:
            self.landsat_data = None  # Setting Landsat raster data to None if not provided

    # Method to generate tiles covering the study area
    def generate_tiles(self):
        # Initialize an empty list to store tiles
        tiles = []
        # Iterate over rows and columns to create tiles
        for i in range(self.num_tiles):
            for j in range(self.num_tiles):
                # Calculate coordinates for each tile
                tile_north = self.north - i * self.tile_height
                tile_south = self.north - (i + 1) * self.tile_height
                tile_east = self.west + (j + 1) * self.tile_width
                tile_west = self.west + j * self.tile_width
                # Create a polygon for the tile and append to the list
                tiles.append(Polygon([(tile_west, tile_south), (tile_east, tile_south), (tile_east, tile_north), (tile_west, tile_north)]))
        # Return tiles as a GeoDataFrame
        return gpd.GeoDataFrame(geometry=tiles, crs="epsg:4326")

    # Method to calculate various metrics for each tile (and plotting)
    def calculate_tile_metrics(self):
        # Calculate land use diversity and plot various visualizations
        diversity_grid, max_diversity_tiles = self.calculate_diversity()
        # Call the functions to plot buildings, land use, heatmaps, and histograms
        self.plot_buildings_landuse()
        self.plot_heatmap(diversity_grid)
        self.plot_landsat_heatmap_overlay(diversity_grid)
        self.plot_histograms(max_diversity_tiles)

    # Method to calculate land use diversity for each tile
    def calculate_diversity(self):
        # Initialize an empty grid to store land use diversity values
        diversity_grid = np.zeros((self.num_tiles, self.num_tiles))  # Grid to store land use diversity
        # Initialize variables to track maximum diversity and corresponding tiles
        max_diversity = 0  # Maximum land use diversity
        max_diversity_tiles = []  # Tiles with maximum land use diversity
    
        # Iterate over tiles
        for i, tile in self.tiles.iterrows():
            # Find land use polygons intersecting with the tile
            land_use_in_tile = self.gdf_land_use[self.gdf_land_use.intersects(tile['geometry'])]  # Land use polygons in the tile
            # Calculate the number of unique land use types in the tile
            land_use_diversity = len(land_use_in_tile['landuse'].unique())  # Number of unique land use types
    
            # Update maximum diversity and corresponding tiles
            if land_use_diversity > max_diversity:
                max_diversity = land_use_diversity
                max_diversity_tiles = [(tile['geometry'].centroid.x, tile['geometry'].centroid.y, land_use_in_tile['landuse'])]  # Updating tiles with maximum diversity
            elif land_use_diversity == max_diversity:
                max_diversity_tiles.append((tile['geometry'].centroid.x, tile['geometry'].centroid.y, land_use_in_tile['landuse']))  # Updating tiles with maximum diversity
    
            # Convert linear index to row and column indices to create heatmap grid (filling w/colors in order)
            row_index = int(i / self.num_tiles)  # Row index
            col_index = i % self.num_tiles  # Column index
            col_index_right_aligned = self.num_tiles - 1 - col_index  # Right-aligned column index
            # Update diversity grid with land use diversity value
            diversity_grid[row_index, col_index_right_aligned] = land_use_diversity  # Updating diversity grid
    
        # Return diversity grid and tiles with maximum diversity
        return diversity_grid, max_diversity_tiles

    # Method to plot buildings, land use, and tiles with land use diversity values
    def plot_buildings_landuse(self):
        # Create a plot with buildings, land use, and tiles
        fig, ax = plt.subplots(figsize=(10, 10))  # Creating a plot
        self.gdf_buildings.plot(ax=ax, alpha=0.5)  # Plotting buildings
        self.gdf_land_use.plot(ax=ax, column='landuse', legend=True, legend_kwds={'title':'Land Use','bbox_to_anchor': (1.05, 0.5), 'loc': 'center left'})  # Plotting land use
        self.tiles.plot(ax=ax, alpha=0.5, edgecolor='blue', facecolor='none')  # Plotting tiles
        # Annotate tiles with land use diversity values
        for i, tile in self.tiles.iterrows():
            land_use_in_tile = self.gdf_land_use[self.gdf_land_use.intersects(tile['geometry'])]  # Land use polygons in the tile
            land_use_diversity = len(land_use_in_tile['landuse'].unique())  # Number of unique land use types
            tile_center = tile['geometry'].centroid  # Tile center
            ax.text(tile_center.x, tile_center.y, str(land_use_diversity), ha='center', va='center', fontsize=10)  # Annotating tiles with diversity values
        ax.set_title('Building Footprints, Land Use, and Tiles')  # Setting title
        plt.show()  # Displaying the plot

    # Method to plot a heatmap of land use diversity
    def plot_heatmap(self, diversity_grid):
        # Define extent of the plot
        xmin, ymin, xmax, ymax = self.tiles.total_bounds  # Extent of the plot
        extent = [xmin, xmax, ymin, ymax]  # Extent of the plot

        # Create a plot with GeoDataFrame layers and heatmap
        fig, ax = plt.subplots(figsize=(10, 10))  # Creating a plot
        heatmap = ax.imshow(diversity_grid, cmap='viridis', interpolation='nearest', alpha=0.7, extent=extent)  # Plotting heatmap
        self.gdf_buildings.plot(ax=ax, color='red', alpha=0.5)  # Plotting buildings
        self.gdf_land_use.plot(ax=ax, column='landuse', legend=True, legend_kwds={'title':'Land Use'}, alpha=0.5)  # Plotting land use
        cbar = plt.colorbar(heatmap)  # Adding colorbar
        cbar.set_label('Land Use Diversity')  # Setting colorbar label
        ax.set_title('Land Use Diversity Heatmap with Buildings and Land Use')  # Setting title
        plt.show()  # Displaying the plot

    # Method to plot Landsat raster overlayed with land use diversity heatmap
    def plot_landsat_heatmap_overlay(self, diversity_grid):
        if self.landsat_data is not None:
            # Create a plot with Landsat raster and land use diversity heatmap
            extent_polygon = Polygon.from_bounds(*self.tiles.total_bounds)  # Extent polygon
            extent_gdf = gpd.GeoDataFrame(geometry=[extent_polygon], crs=self.tiles.crs)  # Extent GeoDataFrame
            extent_gdf = extent_gdf.to_crs(epsg=4326)  # Reprojecting extent GeoDataFrame / to visualize raster and heatmap together
    
            xmin, ymin, xmax, ymax = extent_gdf.geometry.total_bounds  # Extent of the plot
            extent = [xmin, xmax, ymin, ymax]  # Extent of the plot
    
            fig, ax = plt.subplots(figsize=(10, 10))  # Creating a plot
            show(self.landsat_data, ax=ax, extent=extent, cmap='gray')  # Plotting Landsat raster
            heatmap = ax.imshow(diversity_grid, cmap='viridis', interpolation='nearest', alpha=0.7, extent=extent)  # Plotting heatmap
            cbar = plt.colorbar(heatmap)  # Adding colorbar
            cbar.set_label('Land Use Diversity')  # Setting colorbar label
            ax.set_title('Land Use Diversity Heatmap overlaid on Landsat Raster')  # Setting title
            plt.show()  # Displaying the plot
        else:
            print("Landsat raster data is not available. Skipping Landsat raster overlay.")

        
    # Method to plot histograms of land use types for tiles with maximum land use diversity
    def plot_histograms(self, max_diversity_tiles):
        # Iterate over tiles with maximum land use diversity and plot histograms
        for idx, (x, y, land_use_in_tile) in enumerate(max_diversity_tiles):
            plt.figure(figsize=(8, 6))  # Creating a plot
            unique_land_uses, counts = np.unique(land_use_in_tile, return_counts=True)  # Unique land use types and their counts
            
            # Sort land use types and counts in descending order (to visualize in histogram)
            sorted_indices = np.argsort(-counts)  # Sorting indices
            unique_land_uses = unique_land_uses[sorted_indices]  # Sorting land use types
            counts = counts[sorted_indices]  # Sorting counts
            
            # Define a color palette for land use types
            cmap = plt.colormaps.get_cmap('tab20')
            color_palette = cmap(range(len(unique_land_uses)))  # Color palette
            
            for i, (land_use_type, count) in enumerate(zip(unique_land_uses, counts)):
                # Get a color from the color palette
                color = color_palette[i]  # Getting color
                plt.bar(land_use_type, count, color=color, alpha=0.7, label=land_use_type)  # Plotting histogram bars
            plt.title(f'Land Use Diversity at Tile ({x:.2f}, {y:.2f})')  # Setting title
            plt.xlabel('Land Use Type')  # Setting x-label
            plt.ylabel('Frequency')  # Setting y-label
            plt.xticks(rotation=45)  # Rotating x-axis labels
            plt.legend()  # Adding legend
            plt.grid(axis='y', alpha=0.75)  # Adding grid
            plt.show()  # Displaying the plot

# Instantiate LandUseDiversityHM class with study area boundaries and number of tiles
urban_analysis_newyork = LandUseDiversityHM(40.75, 40.70, -74.00, -73.95, num_tiles=10)  # Instantiating LandUseDiversityHM class
# Acquire data including buildings, land use, and Landsat imagery
urban_analysis_newyork.acquire_data("clip_vector.tif") # Acquiring data
# Calculate and visualize various metrics for each tile
urban_analysis_newyork.calculate_tile_metrics()  # Calculating and visualizing metrics


# # Instantiate LandUseDiversityHM class with Ankara city center boundaries and number of tiles
# urban_analysis_ankara = LandUseDiversityHM(39.9500, 39.8750, 32.8675, 32.7875, num_tiles=20)
# # Acquire data including buildings, land use, and Landsat imagery
# urban_analysis_ankara.acquire_data() # Acquiring data
# # Calculate and visualize various metrics for each tile
# urban_analysis_ankara.calculate_tile_metrics()  # Calculating and visualizing metrics