# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(dplyr)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
base_data_regions <- dkuReadDataset("base_data_regions", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
colnames(base_data_regions)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
head(base_data_regions$VUL_vulnerable_groups)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Creating a function that generates a counterfactual dataset

#' @title counterfactual_gen
#' @description Function takes argguments of df, tc, matches and returns a
#' counterfactual dataframe to be used by the hurdle function
#' @param df counterfactual_test_data
#' @param tc tropical cyclone name
#' @param matches municipality matches NOT NEEDED
#' @return counterfactual_data_list return

counterfactual_gen  <- function(df, tc, storm_surge, landslide){
    # function requires dplyr to run correctly
    library(dplyr)

    # get unique municipality codes
    mun_code  <- unique(df$Mun_Code)

    # filter df by tc and get hazard characheristics
    counterfactual_data  <- df %>%
        filter(typhoon == tc) %>% # keep the minimum distance from the filter
        mutate(
              # Primary hazard measurements are based on the closest path to land/municipality.
              # ie where was haddest hit
              HAZ_rainfall_Total = HAZ_rainfall_Total[which.min(HAZ_dis_track_min)],
              HAZ_rainfall_max_6h = HAZ_rainfall_max_6h[which.min(HAZ_dis_track_min)],
              HAZ_rainfall_max_24h = HAZ_rainfall_max_24h[which.min(HAZ_dis_track_min)],
              HAZ_v_max = HAZ_v_max[which.min(HAZ_dis_track_min)],
              HAZ_dis_track_min = min(HAZ_dis_track_min, na.rm = TRUE),
              # setting secondary hard to storm_surge value or landslide value specified in the function call
              GEN_landslide_per = landslide, # fraction of municipality at risk of landslides
              GEN_stormsurge_per = storm_surge, # fraction of municipality at risk of storm surge. 
              GEN_Bu_p_inSSA = storm_surge,
              GEN_Bu_p_LS = landslide,
              GEN_Red_per_LSbldg = landslide,
              GEN_Or_per_LSblg = landslide,
              GEN_Yel_per_LSSAb = storm_surge,
              GEN_RED_per_SSAbldg = storm_surge,
              GEN_OR_per_SSAbldg = storm_surge,
              GEN_Yellow_per_LSbl = landslide,
              TOP_mean_slope = 0.0,
              TOP_mean_elevation_m = 0.0,
              TOP_ruggedness_stdev = 0.0,
              TOP_mean_ruggedness = 0.0,
              TOP_slope_stdev = 0.0,
              VUL_poverty_perc = 0.0,
              GEN_with_coast = 0, 
              GEN_coast_length = 0, 
              VUL_Housing_Units = 0,
              VUL_vulnerable_groups = 0.0, 
              VUL_pantawid_pamilya_beneficiary = 0.0
              ) %>%
        select(-typhoon)

    # which municipalities are not in the filtered data?
    missing_mun  <- setdiff(mun_code, counterfactual_data$Mun_Code)

    # debugging
    #cat("number of missing municipalities:", sep = " ", length(missing_mun))

    # Check if there are any missing municipalities
    if (length(missing_mun) > 0) {
        # Get the characteristics of the missing mun codes
        #remaining_mun <- df %>%
        #    filter(Mun_Code %in% missing_mun) %>%
        #    select(-typhoon, -rain_total, -wind_max, -track_min_dist)

        # Assign the hazard characteristics from the counterfactual data
        remaining_mun <- df %>%
            filter(Mun_Code %in% missing_mun) %>% # after filtering Mun_Code has duplicates how do we remove duplicates?
            distinct(Mun_Code, .keep_all = TRUE) %>%  # Keeps the first occurrence of each Mun_Code
            mutate(HAZ_rainfall_Total = unique(counterfactual_data$HAZ_rainfall_Total),
                  HAZ_rainfall_max_6h = unique(counterfactual_data$HAZ_rainfall_max_6h),
                  HAZ_rainfall_max_24h = unique(counterfactual_data$HAZ_rainfall_max_24h), 
                  HAZ_v_max = unique(counterfactual_data$HAZ_v_max),
                  HAZ_dis_track_min = unique(counterfactual_data$HAZ_dis_track_min),
                  GEN_landslide_per = landslide, # fraction of municipality at risk of landslides
                  GEN_stormsurge_per = storm_surge, # fraction of municipality at risk of storm surge. 
                  GEN_Bu_p_inSSA = storm_surge,
                  GEN_Bu_p_LS = landslide,
                  GEN_Red_per_LSbldg = landslide,
                  GEN_Or_per_LSblg = landslide,
                  GEN_Yel_per_LSSAb = storm_surge,
                  GEN_RED_per_SSAbldg = storm_surge,
                  GEN_OR_per_SSAbldg = storm_surge,
                  GEN_Yellow_per_LSbl = landslide,
                  TOP_mean_slope = 0.0,
                  TOP_mean_elevation_m = 0.0,
                  TOP_ruggedness_stdev = 0.0,
                  TOP_mean_ruggedness = 0.0,
                  TOP_slope_stdev = 0.0,
                  VUL_poverty_perc = 0.0,
                  GEN_with_coast = 0, 
                  GEN_coast_length = 0, 
                  VUL_Housing_Units = 0,
                  VUL_vulnerable_groups = 0.0, 
                  VUL_pantawid_pamilya_beneficiary = 0.0
                  ) %>%
        select(-typhoon)
        # debugging
        cat("number of columns in remaining_mun", sep = " ", ncol(remaining_mun))

        cat("\n number of columns in counterfactual data", sep = " ", ncol(counterfactual_data))

        # Add the remaining municipalities back into the counterfactual data
        counterfactual_data <- rbind(counterfactual_data, remaining_mun)
        
        # remove columns we do not need:
        counterfactual_data  <- counterfactual_data %>%
            select(-c(DAM_perc_dmg,
                      Mun_Code_2,
                      Unnamed..0,
                      Municipality,
                      X10.Digit.Code, 
                      Correspondence.Code, Income.Class,
                      Population.2020.Census., 
                      region,
                      island_groups))
    }

    # df should have all the 1478 municipalities

    return(counterfactual_data) # returns a dataframe (maybe list for more experiments)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
melor_2015  <- counterfactual_gen(df = base_data_regions,
                                  tc = "melor2015",
                                  storm_surge = 0,
                                  landslide =0)

head(melor_2015)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
colnames(melor_2015)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create a list with counterfactual dataframes for each of the tropical cyclones
# Counterfactual dataframes have secondary hazards fixed to 0.0

# extracting all tropical cylone names in the data
tc_names  <- unique(base_data_regions$typhoon)

tc_counterfactual_dfs  <- list()

# loof throught the tc_names to derive counterfactual dataframes

for (tc in tc_names) {
    tc_counterfactual_dfs[[tc]]  <- counterfactual_gen(df = base_data_regions,
                                  tc = tc,
                                  storm_surge = 0,
                                  landslide =0)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
names(tc_counterfactual_dfs)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
v510_counterfactual_datasets_fixed_sec_hazards <- dkuManagedFolderPath("F8NXOAoc")

# Loop through the named list and write each data frame to a CSV
for (name in names(tc_counterfactual_dfs)) {
  df <- tc_counterfactual_dfs[[name]]
  file_path <- file.path(v510_counterfactual_datasets_fixed_sec_hazards, paste0(name, ".csv"))
  write.csv(df, file = file_path, row.names = FALSE)
}