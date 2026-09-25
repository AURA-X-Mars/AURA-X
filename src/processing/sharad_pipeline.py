import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage

class SHARADRadarPipeline:
    DIELECTRIC_CONSTANTS = {
        "VACUUM_CAVITY": 1.0,
        "WATER_ICE": 3.15,
        "REGOLITH_UNCOMPACTED": 2.5,
        "BASALTIC_ROCK": 6.0,
    }

    SPEED_OF_LIGHT = 299792458.0

    def __init__(self, sample_rate_mhz: float = 37.5):
        self.sample_rate = sample_rate_mhz * 1e6
        print(f"[AURA-X SHARAD] Initialized Subsurface Sounding Engine ({sample_rate_mhz} MHz Bandwidth)")

    def deconvolve_chirp_pulse(self, raw_radargram: np.ndarray) -> np.ndarray:
        t = np.linspace(0, 85e-6, int(85e-6 * self.sample_rate))
        reference_chirp = signal.chirp(t, f0=15e6, f1=25e6, t1=85e-6, method='linear')

        compressed_radargram = np.zeros_like(raw_radargram)
        for col in range(raw_radargram.shape[1]):
            compressed = signal.correlate(raw_radargram[:, col], reference_chirp, mode='same')
            compressed_radargram[:, col] = np.abs(compressed)

        radargram_db = 20.0 * np.log10(compressed_radargram + 1e-6)
        return radargram_db

    def detect_subsurface_cavities_and_ice(self, radargram_db: np.ndarray, time_resolution_ns: float = 37.5) -> dict:
        n_depth_samples, n_traces = radargram_db.shape

        surface_indices = np.argmax(radargram_db, axis=0)

        subsurface_mask = np.zeros_like(radargram_db, dtype=bool)
        subsurface_depths_meters = np.zeros(n_traces)
        cavity_detected = np.zeros(n_traces, dtype=bool)

        for col in range(n_traces):
            surf_idx = surface_indices[col]
            if surf_idx + 10 < n_depth_samples:
                subsurface_profile = radargram_db[surf_idx + 10:, col]
                peak_sub_idx = np.argmax(subsurface_profile) + (surf_idx + 10)

                time_delay_sec = (peak_sub_idx - surf_idx) * (time_resolution_ns * 1e-9)
                
                v_basalt = self.SPEED_OF_LIGHT / np.sqrt(self.DIELECTRIC_CONSTANTS["BASALTIC_ROCK"])
                depth_meters = (v_basalt * time_delay_sec) / 2.0
                
                subsurface_depths_meters[col] = depth_meters
                
                if 10.0 <= depth_meters <= 45.0 and radargram_db[peak_sub_idx, col] > -20.0:
                    cavity_detected[col] = True

        return {
            "surface_horizon_indices": surface_indices,
            "calculated_depths_meters": subsurface_depths_meters,
            "lava_tube_cavity_detected": cavity_detected,
            "detected_cavities_count": int(np.sum(cavity_detected)),
            "mean_subsurface_cavity_depth_m": float(np.mean(subsurface_depths_meters[cavity_detected])) if np.any(cavity_detected) else 0.0
        }

if __name__ == "__main__":
    print("--- Running SHARAD Subsurface Radar Pipeline Test ---")

    sharad = SHARADRadarPipeline(sample_rate_mhz=37.5)

    mock_raw_rf_data = np.random.normal(0, 1, (500, 100))
    mock_raw_rf_data[150:160, 20:60] += 5.0

    radargram_db = sharad.deconvolve_chirp_pulse(mock_raw_rf_data)

    radar_results = sharad.detect_subsurface_cavities_and_ice(radargram_db)

    print(f"Radargram Compressed Shape: {radargram_db.shape} (Depth x Orbit Traces)")
    print(f"Subsurface Cavities/Lava Tubes Detected: {radar_results['detected_cavities_count']} locations")
    if radar_results['detected_cavities_count'] > 0:
        print(f"Mean Lava Tube Ceiling Depth:             {radar_results['mean_subsurface_cavity_depth_m']:.2f} meters")