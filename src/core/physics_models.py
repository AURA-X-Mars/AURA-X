import numpy as np

class RadiationShieldingPhysics:
    MU_OVER_RHO_REGOLITH = 0.028  
    
    RHO_REGOLITH = 1.6  
    
    SURFACE_ANNUAL_DOSE_MSV = 230.0  
    
    SAFE_DOSE_LIMIT_MSV = 50.0

    @classmethod
    def calculate_attenuation(cls, depth_meters: float, rock_density_g_cm3: float = 1.6) -> dict:
        depth_cm = depth_meters * 100.0

        areal_density = depth_cm * rock_density_g_cm3

        attenuation_factor = np.exp(-cls.MU_OVER_RHO_REGOLITH * areal_density)
        subsurface_dose_msv = cls.SURFACE_ANNUAL_DOSE_MSV * attenuation_factor

        shielding_efficiency_percent = (1.0 - attenuation_factor) * 100.0
        is_safe_for_human_habitat = subsurface_dose_msv <= cls.SAFE_DOSE_LIMIT_MSV

        return {
            "depth_meters": depth_meters,
            "areal_density_g_cm2": float(round(areal_density, 2)),
            "annual_dose_msv": float(round(subsurface_dose_msv, 2)),
            "shielding_efficiency_percent": float(round(shielding_efficiency_percent, 2)),
            "is_human_safe": bool(is_safe_for_human_habitat)
        }

class ThermalDiffusionPhysics:
    THERMAL_DIFFUSIVITY = 1.0e-7  

    MARS_SOL_SECONDS = 88775.0  

    @classmethod
    def calculate_subsurface_temperature(
        cls, 
        depth_meters: float, 
        mean_surface_temp_c: float = -60.0, 
        temp_amplitude_c: float = 40.0, 
        time_seconds: float = 0.0
    ) -> float:
        omega = (2.0 * np.pi) / cls.MARS_SOL_SECONDS
        skin_depth = np.sqrt((2.0 * cls.THERMAL_DIFFUSIVITY) / omega)

        amplitude_at_depth = temp_amplitude_c * np.exp(-depth_meters / skin_depth)
        phase_shift = depth_meters / skin_depth
        
        temp_c = mean_surface_temp_c + amplitude_at_depth * np.cos(omega * time_seconds - phase_shift)
        return float(round(temp_c, 2))

class GeotechnicalMechanics:
    MARS_GRAVITY = 3.721

    @classmethod
    def calculate_bearing_capacity(
        cls, 
        cohesion_kpa: float = 2.5, 
        friction_angle_deg: float = 35.0, 
        footing_width_meters: float = 5.0,
        bulk_density_g_cm3: float = 1.5
    ) -> float:
        density_kg_m3 = bulk_density_g_cm3 * 1000.0
        phi_rad = np.radians(friction_angle_deg)
        
        N_q = np.tan(np.pi / 4.0 + phi_rad / 2.0)**2 * np.exp(np.pi * np.tan(phi_rad))
        N_c = (N_q - 1.0) / np.tan(phi_rad) if friction_angle_deg > 0 else 5.7
        N_gamma = 2.0 * (N_q + 1.0) * np.tan(phi_rad)
        
        gamma_martian = density_kg_m3 * cls.MARS_GRAVITY
        
        q_ult_pa = (cohesion_kpa * 1000.0 * N_c) + (0.5 * gamma_martian * footing_width_meters * N_gamma)
        q_ult_kpa = q_ult_pa / 1000.0
        
        return float(round(q_ult_kpa, 2))

if __name__ == "__main__":
    print("--- Running Core Physics Engine Validations ---")

    rad_test = RadiationShieldingPhysics.calculate_attenuation(depth_meters=3.0)
    print(f"Radiation Shielding (3m Depth): {rad_test}")

    temp_test = ThermalDiffusionPhysics.calculate_subsurface_temperature(depth_meters=1.0)
    print(f"Subsurface Temp (1m Depth):    {temp_test} °C")
    
    bearing_test = GeotechnicalMechanics.calculate_bearing_capacity()
    print(f"Ultimate Bearing Capacity:      {bearing_test} kPa")