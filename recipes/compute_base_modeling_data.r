library(dataiku)
library(dplyr)

# Recipe inputs
base_inc_data <- dkuReadDataset("base_inc_data", samplingMethod="head", nbRows=100000)

# Compute recipe outputs from inputs
## Renaming columns
base_modeling_data <- base_inc_data %>%
    rename(
    rain_total = HAZ_rainfall_Total,
    rain_max6h = HAZ_rainfall_max_6h,
    rain_max24h = HAZ_rainfall_max_24h,
    wind_max = HAZ_v_max,
    track_min_dist = HAZ_dis_track_min,
    ls_risk_pct = GEN_landslide_per,
    ss_risk_pct = GEN_stormsurge_per,
    blue_ss_frac = GEN_Bu_p_inSSA,
    blue_ls_frac = GEN_Bu_p_LS,
    red_ls_frac = GEN_Red_per_LSbldg,
    orange_ls_frac = GEN_Or_per_LSblg,
    yellow_ss_frac = GEN_Yel_per_LSSAb,
    red_ss_frac = GEN_RED_per_SSAbldg,
    orange_ss_frac = GEN_OR_per_SSAbldg,
    yellow_ls_frac = GEN_Yellow_per_LSbl, # this variable naming was inconsistent, that was annoying
    slope_mean = TOP_mean_slope,
    elev_mean = TOP_mean_elevation_m,
    ruggedness_sd = TOP_ruggedness_stdev,
    ruggedness_mean = TOP_mean_ruggedness,
    slope_sd = TOP_slope_stdev,
    has_coast = GEN_with_coast,
    coast_length = GEN_coast_length,
    poverty_pct = VUL_poverty_perc,
    housing_units = VUL_Housing_Units,
    roof_strong_wall_strong = VUL_StrongRoof_StrongWall,
    roof_strong_wall_light = VUL_StrongRoof_LightWall,
    roof_strong_wall_salv = VUL_StrongRoof_SalvageWall,
    roof_light_wall_strong = VUL_LightRoof_StrongWall,
    roof_light_wall_light = VUL_LightRoof_LightWall,
    roof_light_wall_salv = VUL_LightRoof_SalvageWall,
    roof_salv_wall_strong = VUL_SalvagedRoof_StrongWall,
    roof_salv_wall_light = VUL_SalvagedRoof_LightWall,
    roof_salv_wall_salv = VUL_SalvagedRoof_SalvageWall,
    vulnerable_groups = VUL_vulnerable_groups,
    pantawid_benef = VUL_pantawid_pamilya_beneficiary,
    DAM_perc_dmg = damage_perc     
  )

## Keeping only columns needed for modeling
base_modeling_data <- base_modeling_data %>%
    select(-Mun_code,
           -typhoon,
           -Mun_code2,
           -Unnamed..0,
           -X10.Digit.Code,
           Correspondence.Code,
           -vulnerable_groups, 
           -pantawid_benef, 
           -rain_max6h, 
           -rain_max24h, 
           -poverty_pct,
           -housing_units,
           -Municipality,
           -Income.Class,
           -Population.2020.Census.
          )


# Recipe outputs
dkuWriteDataset(base_modeling_data,"base_modeling_data")
