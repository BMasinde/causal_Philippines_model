# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(stats) # need this to calculate Mahalanobis Distance
library(parallel) # parallelize
library(dplyr)
library(FNN)
library(cluster)
library(ggplot2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs

# Reading the data we need to generate counterfactuals
counterfactual_test_data <- dkuReadDataset("counterfactual_test_data", samplingMethod="head", nbRows=100000)


# path to hardle models and functions
hurdle_components_path <- dkuManagedFolderPath("5NPBmWH1")

# read all models and functions as a list

# List all .rds files in the folder
rds_files <- list.files(hurdle_components_path, pattern = "\\.rds$", full.names = TRUE)

# Read all .rds files into a list
models_n_functions_list <- lapply(rds_files, readRDS)

# Print the names of the loaded objects
names(models_n_functions_list) <- basename(rds_files)

# Display the list
print(models_n_functions_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# sanity check: is the read hurdle_function an actual function?
class(models_n_functions_list$hurdle_function)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
colnames(counterfactual_test_data)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get unique municipality observations
mun_properties  <- counterfactual_test_data %>%
    distinct(Mun_Code,
             blue_ss_frac,
             blue_ls_frac,
             red_ls_frac,
             orange_ls_frac,
             yellow_ss_frac,
             red_ss_frac,
             orange_ss_frac,
             yellow_ls_frac,
             roof_strong_wall_strong,
             roof_strong_wall_light,
             roof_strong_wall_salv,
             roof_light_wall_strong,
             roof_light_wall_light,
             roof_light_wall_salv,
             roof_salv_wall_strong,
             roof_salv_wall_light,
             roof_salv_wall_salv,
             island_groups,
             .keep_all = FALSE)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# variables I'm interested in for matching:
match_vars  <- c('blue_ss_frac',
                    'blue_ls_frac',
                    'red_ls_frac',
                    'orange_ls_frac',
                    'yellow_ss_frac',
                    'red_ss_frac',
                    'orange_ss_frac',
                    'yellow_ls_frac',
                    'roof_strong_wall_strong',
                    'roof_strong_wall_light',
                    'roof_strong_wall_salv',
                    'roof_light_wall_strong',
                    'roof_light_wall_light',
                    'roof_light_wall_salv',
                    'roof_salv_wall_strong',
                    'roof_salv_wall_light',
                    'roof_salv_wall_salv'
                   )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Normalize the variables using z-score
mun_scaled <- mun_properties %>%
  mutate(across(c(blue_ss_frac:roof_salv_wall_salv), scale))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
head(mun_scaled)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Add a column for group labels (if not already present)
# Assuming `group` column exists and contains 3 groups (1, 2, 3)

# Split dataset by group
group1 <- mun_scaled %>% filter(island_groups == "Luzon")
group2 <- mun_scaled %>% filter(island_groups == "Visayas")
group3 <- mun_scaled %>% filter(island_groups == "Mindanao")

# Ensure only numeric columns are used for matching
group1_data <- group1 %>% select(-Mun_Code, -island_groups)
group2_data <- group2 %>% select(-Mun_Code, -island_groups)
group3_data <- group3 %>% select(-Mun_Code, -island_groups)

# Find nearest neighbors from group1 to group2
nn1_2 <- get.knnx(data = group2_data, query = group1_data, k = 1)

# Find nearest neighbors from group1 to group3
nn1_3 <- get.knnx(data = group3_data, query = group1_data, k = 1)

# Merge results
matches <- data.frame(
  Group1_ID = group1$Mun_Code,
  Closest_Group2_ID = group2$Mun_Code[nn1_2$nn.index[,1]],
  Closest_Group3_ID = group3$Mun_Code[nn1_3$nn.index[,1]]
)

# Print results
print(matches)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nrow(matches)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check group sizes
cat("Group Sizes: Luzon =", nrow(group1), " Visayas =", nrow(group2), " Mindanao =", nrow(group3), "\n")

# Ensure only numeric columns are used
group1_data <- group1 %>% select(-Mun_Code, -island_groups)
group2_data <- group2 %>% select(-Mun_Code, -island_groups)
group3_data <- group3 %>% select(-Mun_Code, -island_groups)

# Find nearest neighbors
nn1_2 <- get.knnx(data = group2_data, query = group1_data, k = 1)
nn1_3 <- get.knnx(data = group3_data, query = group1_data, k = 1)

# Ensure indices do not exceed available rows
nn1_2_indices <- pmin(nn1_2$nn.index[,1], nrow(group2))
nn1_3_indices <- pmin(nn1_3$nn.index[,1], nrow(group3))

# Merge results
matches <- data.frame(
  Group1_ID = group1$Mun_Code,
  Closest_Group2_ID = group2$Mun_Code[nn1_2_indices],
  Closest_Group3_ID = group3$Mun_Code[nn1_3_indices]
)

# Print results
print(dim(matches))  # Should match group1's size
print(head(matches))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ATTEMPT three at finding similar municipalities
# library(dplyr)
# library(cluster)

# # Combine the datasets
# all_data <- bind_rows(
#   group1 %>% mutate(region = "Luzon"),
#   group2 %>% mutate(region = "Visayas"),
#   group3 %>% mutate(region = "Mindanao")
# )

# # Remove non-numeric columns
# all_numeric <- all_data %>% select(-Mun_Code, -island_groups, -region)

# # Perform clustering (adjust k based on your dataset size)
# set.seed(123)  # For reproducibility
# k <- 10  # Adjust this to define the number of groups
# clusters <- kmeans(all_numeric, centers = k, nstart = 25)

# # Add cluster assignments back to the data
# all_data$Cluster <- clusters$cluster

# # Split the data back into groups
# clustered_groups <- all_data %>%
#   group_by(Cluster) %>%
#   summarise(
#     Luzon = list(Mun_Code[region == "Luzon"]),
#     Visayas = list(Mun_Code[region == "Visayas"]),
#     Mindanao = list(Mun_Code[region == "Mindanao"])
#   )

# # Print out clustered groups
# print(clustered_groups)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ATTEMPT three at finding similar municipalities

# Combine the datasets
all_data <- bind_rows(
  group1 %>% mutate(region = "Luzon"),
  group2 %>% mutate(region = "Visayas"),
  group3 %>% mutate(region = "Mindanao")
)

# Remove non-numeric columns except for Mun_Code and region
all_numeric <- all_data %>% select(-Mun_Code, -island_groups, -region)

# Perform clustering
set.seed(123)  # For reproducibility
k <- 10  # Number of clusters (adjust as needed)
clusters <- kmeans(all_numeric, centers = k, nstart = 25)

# Add cluster assignments back to the data
all_data$Cluster <- clusters$cluster

# Create a tibble summarizing cluster sizes and municipality codes
cluster_summary <- all_data %>%
  group_by(Cluster) %>%
  summarise(
    Luzon = list(Mun_Code[region == "Luzon"]),
    Visayas = list(Mun_Code[region == "Visayas"]),
    Mindanao = list(Mun_Code[region == "Mindanao"])
  )

# Convert tibble into a nested list containing municipality codes
nested_list <- cluster_summary %>%
  mutate(Cluster = as.character(Cluster)) %>%  # Convert Cluster to character for list keys
  split(.$Cluster) %>%
  lapply(function(row) {
    list(
      Luzon = row$Luzon[[1]],
      Visayas = row$Visayas[[1]],
      Mindanao = row$Mindanao[[1]]
    )
  })

# Print outputs
print(cluster_summary)  # Summarized tibble with Mun_Code
print(nested_list)  # Nested list with Mun_Code

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # A DIFFERENT WAY OF LISTING
# # Combine the datasets
# all_data <- bind_rows(
#   group1 %>% mutate(region = "Luzon"),
#   group2 %>% mutate(region = "Visayas"),
#   group3 %>% mutate(region = "Mindanao")
# )

# # Remove non-numeric columns except for Mun_Code and region
# all_numeric <- all_data %>% select(-Mun_Code, -island_groups, -region)

# # Perform clustering
# set.seed(123)  # For reproducibility
# k <- 10  # Number of clusters (adjust as needed)
# clusters <- kmeans(all_numeric, centers = k, nstart = 25)

# # Add cluster assignments back to the data
# all_data$Cluster <- clusters$cluster

# # Create a tibble summarizing cluster sizes and municipality codes
# cluster_summary <- all_data %>%
#   group_by(Cluster) %>%
#   summarise(
#     Luzon = list(Mun_Code[region == "Luzon"]),
#     Visayas = list(Mun_Code[region == "Visayas"]),
#     Mindanao = list(Mun_Code[region == "Mindanao"])
#   )

# # Convert tibble into a nested list containing municipality codes
# nested_list <- cluster_summary %>%
#   mutate(Cluster = as.character(Cluster)) %>%  # Convert Cluster to character for list keys
#   split(.$Cluster) %>%
#   lapply(function(row) {
#     # Directly return the list of Mun_Code for each region in each cluster
#     c(Luzon = row$Luzon[[1]], Visayas = row$Visayas[[1]], Mindanao = row$Mindanao[[1]])
#   })

# # Print outputs
# print(cluster_summary)  # Summarized tibble with Mun_Code
# print(nested_list)  # Nested list with Mun_Code (no extra nesting)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# some 2nd level lists have no entries
# cleaning up
clean_list <- function(lst) {
  # Recursively clean second-level entries
  lst <- lapply(lst, function(sublist) {
    if (is.list(sublist)) {
      sublist <- clean_list(sublist)  # Recursively clean sublists
      if (length(sublist) == 0) return(NULL)  # Remove empty sublists
    } else if (length(sublist) == 0) {
      return(NULL)  # Remove empty atomic vectors
    }
    return(sublist)
  })
  
  # Remove NULL entries from first-level list
  lst <- lst[!sapply(lst, is.null)]
  
  # Remove first-level entries that have 0 or only 1 non-empty sublist
  lst <- lst[sapply(lst, function(sublist) length(sublist) > 1)]
  
  # If the entire list is empty, return NULL
  if (length(lst) == 0) return(NULL)
  
  return(lst)
}

cleaned_list <- clean_list(nested_list)

print(cleaned_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Creating a function that generates a counterfactual dataset

#' @title counterfactual_gen
#' @description Function takes argguments of df, tc, matches and returns a
#' counterfactual dataframe to be used by the hurdle function
#' @param df counterfactual_test_data
#' @param tc tropical cyclone name
#' @param matches municipality matches NOT NEEDED
#' @return counterfactual_data_list return

counterfactual_gen  <- function(df, tc){

    # get unique municipality codes
    mun_code  <- unique(df$Mun_Code)

    # filter df by tc and get hazard characheristics
    counterfactual_data  <- df %>%
        filter(typhoon == tc) %>% # keep the minimum distance from the filter
        mutate(track_min_dist = min(track_min_dist, na.rm = TRUE),
              rain_total = rain_total[which.min(track_min_dist)],
              wind_max = wind_max[which.min(track_min_dist)],
              wind_blue_ss = wind_max * blue_ss_frac,
              wind_yellow_ss = wind_max * yellow_ss_frac,
              wind_orange_ss = wind_max * orange_ss_frac,
              wind_red_ss = wind_max * red_ss_frac,
              rain_blue_ss = rain_total * blue_ls_frac,
              rain_yellow_ss = rain_total * yellow_ls_frac,
              rain_orange_ss = rain_total * orange_ls_frac,
              rain_red_ss = rain_total * red_ls_frac
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
            mutate(rain_total = unique(counterfactual_data$rain_total),
                   wind_max = unique(counterfactual_data$wind_max),
                   wind_blue_ss = wind_max * blue_ss_frac,
                   wind_yellow_ss = wind_max * yellow_ss_frac,
                   wind_orange_ss = wind_max * orange_ss_frac,
                   wind_red_ss = wind_max * red_ss_frac,
                   rain_blue_ss = rain_total * blue_ls_frac,
                   rain_yellow_ss = rain_total * yellow_ls_frac,
                   rain_orange_ss = rain_total * orange_ls_frac,
                   rain_red_ss = rain_total * red_ls_frac,
                   damage_perc = 0, # set damage variable to zero or "Damage_below_10"
                   damage_binary = 0,
                   damage_binary_2 = "Damage_below_10"
                  ) %>%
        select(-typhoon)
        # debugging
        cat("number of columns in remaining_mun", sep = " ", ncol(remaining_mun))

        cat("\n number of columns in counterfactual data", sep = " ", ncol(counterfactual_data))

        # Add the remaining municipalities back into the counterfactual data
        counterfactual_data <- rbind(counterfactual_data, remaining_mun)
    }

    # df should have all the 1478 municipalities

    return(counterfactual_data) # returns a dataframe (maybe list for more experiments)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
melor_2015  <- counterfactual_gen(df = counterfactual_test_data, tc = "melor2015")

head(melor_2015)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Counterfactual Testing on Typhoon Melor 2015

# extracting the hurdle function from the list of models and functions
hurdle_function  <- models_n_functions_list$hurdle_function

# hurdle fuctions requires:
# @param df the dataframe
# @param base models as a list
# @param high impact models as a list

base_models  <- list(models_n_functions_list$base_clas_full_model,
                     models_n_functions_list$base_rain_model,
                     models_n_functions_list$base_reg_model,
                     models_n_functions_list$base_wind_model
                    )

# makes sure the list has correct names
names(base_models)  <- c("base_clas_full_model","base_rain_model", "base_reg_model", "base_wind_model")


trunc_models  <- list(models_n_functions_list$trunc_rain_model,
                      models_n_functions_list$trunc_reg_model,
                      models_n_functions_list$trunc_wind_model
                    )

# makes sure the list has correct names
names(trunc_models)  <- c("trunc_rain_model","trunc_reg_model", "trunc_wind_model")


counterfactual_hurdle_preds  <- hurdle_function(df = melor_2015,
                                               scm_models_base = base_models,
                                               scm_models_high = trunc_models,
                                               threshold = 0.35 # threshold in train/test models is 0.35
                                               )

# Remember hurdle function returns predictions
# TO DO List to make my work here easier
# remember to set threshold to a default of 0.35
# hurdle function should check if the packages dplyr, rpart and caret are loaded or preload them

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# append the results to the counterfactual dataset
melor_2015  <- melor_2015 %>%
    mutate(damage_preds = counterfactual_hurdle_preds)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# plotting boxplots of the counterfactuals

# Assuming your data frame is named df
# Loop through each entry in the nested list

plots_list  <- list()
means_list  <- list()

for (i in seq_along(cleaned_list)) {
  # Get the current entry
  current_entry <- cleaned_list[[i]]
  
  # Convert the nested list entry to a data frame format
  plot_data <- bind_rows(lapply(names(current_entry), function(region) {
    data.frame(Mun_Code = unlist(current_entry[[region]]), island_regions = region, stringsAsFactors = FALSE)
  }))
  
  # Merge with original data to get predicted damage
  merged_data <- melor_2015 %>%
    inner_join(plot_data, by = "Mun_Code")
  
  # Create boxplot
  p <- ggplot(merged_data, aes(x = island_groups, y = damage_preds, fill = island_groups)) +
    geom_boxplot() +
    labs(title = paste("Predicted Damage Distribution - List Entry", i),
         x = "Island Region",
         y = "Predicted Damage") +
    theme_minimal()
    
 # Save the plot in the list
  plots_list[[i]] <- p
    
  # Calculate the mean of damage_preds for each island_groups
  mean_values <- merged_data %>%
    group_by(island_groups) %>%
    summarise(mean_damage = mean(damage_preds, na.rm = TRUE))
  
  # Save the means in the list
  means_list[[i]] <- mean_values
  
}

# Check the list to confirm plots are stored
print(plots_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
matching_counterfactuals <- dkuManagedFolderPath("ZO3oPxC1")