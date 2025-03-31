# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(dplyr)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
base_data_regions <- dkuReadDataset("base_data_regions", samplingMethod="head", nbRows=100000)

# Compute recipe outputs from inputs
# Renaming and Removing columns we don't need
counterfactual_test_data <- base_data_regions %>%
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
    damage_perc = DAM_perc_dmg
  ) %>%
    select(-Mun_Code_2,
           -Unnamed..0,
           -X10.Digit.Code,
           -Correspondence.Code,
           -vulnerable_groups,
           -pantawid_benef,
           -rain_max6h,
           -rain_max24h,
           -poverty_pct,
           -housing_units,
           -Income.Class,
           -Population.2020.Census.,
           -poverty_pct
          )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Handling NULL values in outcome (damage_perc)
# Update damage_perc column based on conditions
counterfactual_test_data$damage_perc <- with(counterfactual_test_data, {
  # Check if damage_perc is NA and if wind_max is less than 25 and rain_total is less than 50
  ifelse(
    is.na(damage_perc) & wind_max < 25 & rain_total < 50,  # condition to check
    0,  # if condition is true, set damage_perc to 0
    damage_perc  # otherwise, retain the original value of DAM_perc_dmg
  )
})


# Remove observations that remain with NULL values
counterfactual_test_data <- counterfactual_test_data  %>%
  filter(
      !is.na(damage_perc)
  )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Creating a binary outcome (damage_binary) (1 = damage_perc >= 10, 0 otherwise)
counterfactual_test_data$damage_binary <- with(counterfactual_test_data, {
  ifelse(
      damage_perc >= 10, # check if damage_perc is greater or equal to 10
      1, # if condition is true, set damage_binary to 1
      0 # otherwise, set to zero
  )
})

# binary outcome converted to factor
counterfactual_test_data$damage_binary <- factor(counterfactual_test_data$damage_binary)

# including labels to the factors
counterfactual_test_data$damage_binary_2 <- factor(counterfactual_test_data$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# Recipe outputs
dkuWriteDataset(counterfactual_test_data,"counterfactual_test_data")