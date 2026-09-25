import numpy as np

class MarsSpatialTransforms:
    MARS_EQUATORIAL_RADIUS_KM = 3396.19
    MARS_POLAR_RADIUS_KM = 3376.20
    FLATTENING = (MARS_EQUATORIAL_RADIUS_KM - MARS_POLAR_RADIUS_KM) / MARS_EQUATORIAL_RADIUS_KM

    @classmethod
    def planetocentric_to_cartesian(cls, lat_deg: float, lon_deg: float, elevation_m: float = 0.0) -> dict:
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        radius_km = cls.MARS_EQUATORIAL_RADIUS_KM + (elevation_m / 1000.0)
        
        x = radius_km * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius_km * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius_km * np.sin(lat_rad)
        
        return {
            "x_km": float(round(x, 4)),
            "y_km": float(round(y, 4)),
            "z_km": float(round(z, 4)),
            "radius_vector_km": float(round(radius_km, 4))
        }

    @classmethod
    def dem_raster_to_point_cloud(
        cls, 
        dem_matrix: np.ndarray, 
        center_lat: float, 
        center_lon: float, 
        resolution_meters_per_pixel: float = 100.0
    ) -> dict:
        rows, cols = dem_matrix.shape
        
        x_offsets = (np.arange(cols) - (cols / 2.0)) * resolution_meters_per_pixel
        y_offsets = ((rows / 2.0) - np.arange(rows)) * resolution_meters_per_pixel
        
        grid_x, grid_y = np.meshgrid(x_offsets, y_offsets)
        grid_z = dem_matrix.astype(np.float32)

        dz_dx, dz_dy = np.gradient(grid_z, resolution_meters_per_pixel)
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        points_flat = np.column_stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()])
        
        return {
            "vertex_count": int(points_flat.shape[0]),
            "points_array": points_flat,
            "slope_matrix_deg": slope_degrees,
            "bounding_box_meters": {
                "width": float(cols * resolution_meters_per_pixel),
                "height": float(rows * resolution_meters_per_pixel),
                "min_elevation": float(np.min(grid_z)),
                "max_elevation": float(np.max(grid_z))
            }
        }

if __name__ == "__main__":
    print("--- Running Spatial Transforms Core Validation ---")
    
    cartesian = MarsSpatialTransforms.planetocentric_to_cartesian(lat_deg=39.3, lon_deg=189.7, elevation_m=-3000.0)
    print(f"Arcadia Planitia 3D Cartesian Vector: {cartesian}")
    
    mock_dem = np.random.uniform(-3100, -2900, (50, 50))
    mesh = MarsSpatialTransforms.dem_raster_to_point_cloud(mock_dem, center_lat=39.3, center_lon=189.7)
    
    print(f"Generated Mesh Vertices: {mesh['vertex_count']} Points")
    print(f"Max Elevation Delta:     {mesh['bounding_box_meters']['max_elevation']:.2f} m")
    print(f"Mean Slope Angle:        {np.mean(mesh['slope_matrix_deg']):.2f}°")