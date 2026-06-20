import sys

# IS 456:2000 Table 5 | "Exposure condition": [Minimum cement content in kg/m^3, Maximum water to cement ratio]
IS456_T5_R = {
    "Mild": [220, 0.55],
    "Moderate": [300, 0.50],
    "Severe": [320, 0.45],
    "Very severe": [340, 0.45],
    "Extreme": [360, 0.40]
}

IS456_T5_P = {
    "Mild": [300, 0.60],
    "Moderate": [240, 0.60],
    "Severe": [250, 0.50],
    "Very severe": [260, 0.45],
    "Extreme": [280, 0.40]
}

# IS 10262:2009 Table 2 | "Grade": Assumed Standard Deviation
IS10262_T1 = {
    "M1": 3.5,
    "M2": 4.0,
    "M3": 5.0,
    "M4": 6.0
}

# IS 10262:2009 Table 4 | "Nominal Maximum Size Of Aggregate in mm": Maximum Water content in kg
IS10262_T2 = {
    "10": 208,
    "20": 186,
    "40": 165
}

# IS 10262:2009 Table 5 | "Nominal Maximum Size Of Aggregate in mm": (vol of coarse aggregates)
IS10262_T3 = {
    "10": [0.50, 0.48, 0.46, 0.44],
    "20": [0.66, 0.64, 0.62, 0.60],
    "40": [0.75, 0.73, 0.71, 0.69]
}

def target_strength_calculation(grade):
    """Calculate target strength for mix proportioning."""
    grade_mapping = {
        "M10": "M1", "M15": "M1",
        "M20": "M2", "M25": "M2",
        "M30": "M3", "M35": "M3", 
        "M40": "M3", "M45": "M3", 
        "M50": "M3", "M55": "M3", 
        "M60": "M3","M65": "M4", 
        "M70": "M4"
    }
    
    g = grade_mapping.get(grade, "M4")
    sd = IS10262_T1.get(g, 6.0)  # Default to M4 if grade not found
    return int(grade.replace("M", "")) + (1.65 * sd)

def water_cement_ratio_calculation(exposure, type_conc):
    """Select water-cement ratio based on exposure and type of concrete."""
    exposure = exposure.capitalize()
    data = IS456_T5_P if type_conc == "Plain" else IS456_T5_R
    
    if exposure in data:
        return data[exposure][1]
    else:
        raise ValueError(f"Invalid exposure condition: {exposure} for type {type_conc}")

def water_content_calculation(slump, soa, toa, chem_ad):
    """Calculate water content based on slump, size of aggregate, type of aggregate, and chemical admixture."""
    water_content = IS10262_T2.get(soa, 186)  # Default to 20mm if size not found
    
    # Adjust water content based on type of aggregate
    adjustments = {
        "Sub-angular": -10,
        "Gravel": -20,
        "Rounded Gravel": -25
    }
    water_content += adjustments.get(toa, 0)
    
    # Adjust water content for slump > 50mm
    if slump > 50:
        n = (slump - 50) / 25
        water_content += (0.03 * n * water_content)
    
    # Adjust water content for chemical admixture
    if chem_ad == "Super Plasticizer":
        water_content *= 0.8
    elif chem_ad == "Plasticizer":
        water_content *= 0.9
    
    return water_content

def cement_content_calculation(exposure, wcr, wc, type_conc):
    """Calculate cement content based on exposure, water-cement ratio, water content, and type of concrete."""
    exposure = exposure.capitalize()
    data = IS456_T5_P if type_conc == "Plain" else IS456_T5_R
    
    if exposure in data:
        min_cc = data[exposure][0]
        cement_content = wc / wcr
        return max(cement_content, min_cc)
    else:
        raise ValueError(f"Invalid exposure condition: {exposure} for type {type_conc}")

def fly_cement_content_calculation(exposure, wcr, wc, type_conc):
    """Calculate cement and fly ash content."""
    exposure = exposure.capitalize()
    data = IS456_T5_P if type_conc == "Plain" else IS456_T5_R
    
    if exposure in data:
        min_cc = data[exposure][0]
        cement_content = wc / wcr
        temp1 = max(cement_content, min_cc)
        
        cement_content *= 1.10
        new_wcr = wc / cement_content
        flyash_content = cement_content * 0.3
        temp2 = cement_content - flyash_content
        
        if temp2 < 270:
            i = 0.25
            while i > 0:
                temp2 = cement_content - (cement_content * i)
                if temp2 >= 270:
                    print(f"\nPercentage of Fly Ash is {int(i * 100)}%\n")
                    break
                i -= 0.05
            else:
                sys.exit("Mix is not possible!! (Cement < 270)")
        
        cement_content = temp2
        cement_saved = temp1 - cement_content
        return cement_content, flyash_content, cement_saved, new_wcr
    else:
        raise ValueError(f"Invalid exposure condition: {exposure} for type {type_conc}")

def vol_of_CAnFA_calculation(zone, soa, wcr, pumping):
    """Calculate volume of coarse and fine aggregate."""
    zone_index = {"Zone 4": 0, "Zone 3": 1, "Zone 2": 2, "Zone 1": 3}.get(zone, 0)
    vol_CA = IS10262_T3.get(soa, [0.66, 0.64, 0.62, 0.60])[zone_index]
    
    if wcr > 0.5:
        vol_CA -= 0.01 * ((wcr - 0.5) / 0.05)
    else:
        vol_CA += 0.01 * ((0.5 - wcr) / 0.05)
    
    if pumping:
        vol_CA *= 0.9
    
    vol_FA = 1 - vol_CA
    return vol_CA, vol_FA

def mix_calculation(cc, sp_c, wc, v_ca, v_fa, sp_ca, sp_fa, sp_chemad):
    """Calculate mix proportions per unit volume of concrete."""
    vol_cement = (cc / sp_c) * 0.001
    vol_water = wc * 0.001
    mass_of_chemAd = cc * 0.02
    vol_chemAd = (mass_of_chemAd / sp_chemad) * 0.001
    vol_all_aggr = 1 - (vol_cement + vol_water + vol_chemAd)
    
    mass_CA = vol_all_aggr * v_ca * sp_ca * 1000
    mass_FA = vol_all_aggr * v_fa * sp_fa * 1000
    
    return mass_of_chemAd, mass_CA, mass_FA

def fly_mix_calculation(cc, sp_c, wc, v_ca, v_fa, sp_ca, sp_fa, sp_fly, sp_chemad, fc):
    """Calculate mix proportions with fly ash per unit volume of concrete."""
    vol_cement = (cc / sp_c) * 0.001
    vol_flyash = (fc / sp_fly) * 0.001
    vol_water = wc * 0.001
    mass_of_chemAd = cc * 0.02
    vol_chemAd = (mass_of_chemAd / sp_chemad) * 0.001
    vol_all_aggr = 1 - (vol_cement + vol_flyash + vol_water + vol_chemAd)
    
    mass_CA = vol_all_aggr * v_ca * sp_ca * 1000
    mass_FA = vol_all_aggr * v_fa * sp_fa * 1000
    
    return mass_of_chemAd, mass_CA, mass_FA