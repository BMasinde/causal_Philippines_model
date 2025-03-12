# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)

# Recipe inputs
base_data_regions <- dkuReadDataset("base_data_regions", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# Removing columns we don't need
counterfactual_test_data <- base_data_regions  %>%
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
counterfactual_test_data$damage_perc <- with(modeling_data, {
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
counterfactual_test_data$damage_binary_2 <- factor(counterfactual_test_data2$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
dkuWriteDataset(counterfactual_test_data,"counterfactual_test_data")