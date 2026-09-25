import numpy as np
import scipy.ndimage as ndimage

class HiRISEProcessingPipeline:
    def __init__(self, resolution_m_per_px: float = 0.25):
        self.resolution = resolution_m_per_px
        print(f"[AURA-X HiRISE] Initialized High-Res Photogrammetry Engine ({self.resolution} m/pixel)")

    def generate_dem_from_stereo_pair(self, left_ortho: np.ndarray, right_ortho: np.ndarray) -> np.ndarray:
        left_norm = (left_ortho - np.mean(left_ortho)) / (np.std(left_ortho) + 1e-6)
        right_norm = (right_ortho - np.mean(right_ortho)) / (np.std(right_ortho) + 1e-6)

        dx = ndimage.sobel(left_norm, axis=1) - ndimage.sobel(right_norm, axis=1)
        disparity_map = np.abs(dx)

        elevation_dem = ndimage.gaussian_filter(disparity_map * 15.0, sigma=1.0)
        return elevation_dem

    def detect_landing_hazards(self, dem_grid: np.ndarray, boulder_threshold_meters: float = 0.50) -> dict:
        rows, cols = dem_grid.shape

        dz_dx, dz_dy = np.gradient(dem_grid, self.resolution)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        laplacian = ndimage.laplace(dem_grid)
        boulder_mask = np.abs(laplacian) > boulder_threshold_meters

        roughness = ndimage.generic_filter(dem_grid, np.std, size=3)

        safe_zone_mask = (slope_deg < 8.0) & (~boulder_mask) & (roughness < 0.2)

        safe_area_m2 = np.sum(safe_zone_mask) * (self.resolution ** 2)
        total_area_m2 = (rows * cols) * (self.resolution ** 2)

        return {
            "slope_degrees_map": slope_deg,
            "boulder_hazard_mask": boulder_mask,
            "terrain_roughness_map": roughness,
            "safe_landing_mask": safe_zone_mask,
            "hazard_metrics": {
                "max_slope_deg": float(np.max(slope_deg)),
                "mean_slope_deg": float(np.mean(slope_deg)),
                "boulder_count_estimate": int(np.sum(boulder_mask)),
                "safe_landing_area_m2": float(safe_area_m2),
                "safe_landing_area_percentage": float((safe_area_m2 / total_area_m2) * 100.0)
            }
        }

if __name__ == "__main__":
    print("--- Running HiRISE Photogrammetry Pipeline Test ---")

    hirise = HiRISEProcessingPipeline(resolution_m_per_px=0.25)

    mock_left_image = np.random.uniform(0.2, 0.8, (500, 500))
    mock_right_image = mock_left_image + np.random.normal(0, 0.05, (500, 500))

    dem_mesh = hirise.generate_dem_from_stereo_pair(mock_left_image, mock_right_image)

    hazard_analysis = hirise.detect_landing_hazards(dem_mesh)

    print(f"DEM Grid Generated:        {dem_mesh.shape[0]}x{dem_mesh.shape[1]} pixels")
    print(f"Max Surface Slope:         {hazard_analysis['hazard_metrics']['max_slope_deg']:.2f}°")
    print(f"Boulders/Hazards Detected: {hazard_analysis['hazard_metrics']['boulder_count_estimate']} points")
    print(f"Safe Landing Zone Area:    {hazard_analysis['hazard_metrics']['safe_landing_area_m2']:.1f} m² ({hazard_analysis['hazard_metrics']['safe_landing_area_percentage']:.2f}%)")