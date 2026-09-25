import os
import numpy as np
import scipy.ndimage as ndimage

class CRISMSpectralPipeline:
    BAND_WAVELENGTHS_NM = {
        "R1080": 1080.0,
        "R1500": 1500.0,
        "R1900": 1900.0,
        "R2500": 2500.0,
        "R530":   530.0,
    }

    def __init__(self, data_cube_shape=(544, 256, 256)):
        self.bands, self.height, self.width = data_cube_shape
        print(f"[AURA-X CRISM] Initialized Pipeline for Spectral Cube ({self.bands} Bands, {self.height}x{self.width})")

    def calibrate_radiance_to_reflectance(self, raw_cube: np.ndarray, solar_zenith_angle_deg: float = 45.0) -> np.ndarray:
        cos_zenith = np.cos(np.radians(solar_zenith_angle_deg))
        if cos_zenith <= 0:
            cos_zenith = 0.01
            
        reflectance_cube = np.clip(raw_cube / cos_zenith, 0.0, 1.0)
        
        for b in range(reflectance_cube.shape[0]):
            reflectance_cube[b] = ndimage.gaussian_filter(reflectance_cube[b], sigma=0.5)
            
        return reflectance_cube

    def compute_mineral_summary_parameters(self, io_f_cube: np.ndarray) -> dict:
        idx_530  = int((530 - 400) / (3900 - 400) * self.bands)
        idx_1080 = int((1080 - 400) / (3900 - 400) * self.bands)
        idx_1500 = int((1500 - 400) / (3900 - 400) * self.bands)
        idx_1900 = int((1900 - 400) / (3900 - 400) * self.bands)
        idx_2500 = int((2500 - 400) / (3900 - 400) * self.bands)

        r_530  = io_f_cube[idx_530]
        r_1080 = io_f_cube[idx_1080]
        r_1500 = io_f_cube[idx_1500]
        r_1900 = io_f_cube[idx_1900]
        r_2500 = io_f_cube[idx_2500]

        continuum_1900 = 0.6 * r_1500 + 0.4 * r_2500
        bd1900 = 1.0 - np.divide(r_1900, continuum_1900 + 1e-6)
        bd1900 = np.clip(bd1900, 0.0, 1.0)

        feindex = np.divide(r_1080 - r_530, r_1080 + r_530 + 1e-6)

        return {
            "bd1900_water_ice_index": bd1900,
            "ferric_iron_index": feindex,
            "raw_reflectance_1500nm": r_1500,
            "detected_water_ice_pixels_count": int(np.sum(bd1900 > 0.15))
        }

if __name__ == "__main__":
    print("--- Running CRISM Hyperspectral Pipeline Test ---")

    pipeline = CRISMSpectralPipeline(data_cube_shape=(544, 256, 256))

    mock_raw_radiance = np.random.uniform(0.1, 0.8, (544, 256, 256))

    calibrated_reflectance = pipeline.calibrate_radiance_to_reflectance(mock_raw_radiance, solar_zenith_angle_deg=38.5)

    mineral_indices = pipeline.compute_mineral_summary_parameters(calibrated_reflectance)
    
    print(f"Calibrated Reflectance Min/Max: {np.min(calibrated_reflectance):.3f} / {np.max(calibrated_reflectance):.3f}")
    print(f"Water Ice Absorption BD1900 Mean: {np.mean(mineral_indices['bd1900_water_ice_index']):.4f}")
    print(f"Ferric Iron Index Mean:           {np.mean(mineral_indices['ferric_iron_index']):.4f}")
    print(f"High Water-Ice Depots Count:      {mineral_indices['detected_water_ice_pixels_count']} Pixels")