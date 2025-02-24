# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)
library(dplyr)
library(caret)
library(data.table)
library(mlflow)
library(reticulate)
library(Metrics)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
clas_folder_path <- dkuManagedFolderPath("xcPrnvPS")
base_reg_path <- dkuManagedFolderPath("ZijSaAqQ")
trunc_reg_path <- dkuManagedFolderPath("dL4i4SKb")

# Final Testing data --------------------------------------------------------- 
# Reading base_test data
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)

# Redaing trunc_test data
truncated_test <- dkuReadDataset("truncated_test", samplingMethod="head", nbRows=100000)
# ----------------------------------------------------------------------------



# Read the base .rds models --------------------------------------

# reading classification model
base_clas_min_model  <- readRDS(
    file.path(clas_folder_path, "base_clas_min_model.rds")
)

# reading the base wind and rain models
base_wind_model  <- readRDS(
    file.path(base_reg_path, "base_wind_model.rds")
)
base_rain_model  <- readRDS(
    file.path(base_reg_path, "base_rain_model.rds")
)

# base regression model
base_reg_model  <- readRDS(
    file.path(base_reg_path, "base_reg_min_model.rds")
)

# Reading the truncated .rds models -------------------------------

# reading the trunc wind and rain models
trunc_wind_model  <- readRDS(
    file.path(trunc_reg_path, "trunc_wind_model.rds")
)
trunc_rain_model  <- readRDS(
    file.path(trunc_reg_path, "trunc_rain_model.rds")
)

# base regression model
trunc_reg_model  <- readRDS(
    file.path(trunc_reg_path, "trunc_reg_min_model.rds")
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Putting models into lists

## trained SCM models on base data
base_models <- list(
  "wind_model" = base_wind_model,
  "rain_model" = base_rain_model,
  "base_reg_model" = base_reg_model
)

## trained SCM models on high impact data (damage >= 10)

high_models <- list(
  "wind_model_high" = trunc_wind_model,
  "rain_model_high" = trunc_rain_model,
  "high_reg_model" =  trunc_reg_model
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# HURDLE METHOD FUNCTION
#' Title: Predict the building damage % from TCs
#'
#' Function takes the test data & trained models and returns predicted building damages.
#' 
#' @param df A dataframe for prediction (can be the test set for testing hurdle method)
#' @param class_model The trained model for classification
#' @param scm_models_base A list of the SCM models for the base regression 
#' @param scm_models_high A list of SCM models for the high-impact regression
#' 
#' 

predictDamage <- function(df, class_model, scm_models_base, scm_models_high) {
  
  ## common predictions btw class & base regression
  wind_max_base_pred <- predict(scm_models_base$wind_model, 
                         newdata = df)
  
  rain_total_base_pred <- predict(scm_models_base$rain_model, 
                         newdata = df)
  
  ## adding predictions to the df
  df <- df %>%
  mutate(wind_max_pred = wind_max_base_pred, 
         rain_total_pred = rain_total_base_pred
         )
  
  ## Step 1: Predict the class label (whether the damage will exceed the threshold)
  ## class_model should return predicted classes and not probs.
  ## class_model expects variables "wind_max_pred" and "rain_total_pred" in dataframe df
  class_pred <- predict(class_model, df, type = "class")  
  
  ## Step 2: Predict the base damage percentage using the base regression model (for low impact cases)
  ## base expects variables "wind_max_pred" and "rain_total_pred" in dataframe df
  ## should return the predicted damage percentages
  base_pred <- predict(scm_models_base$base_reg_model, df)
  
  ## Step 3: Predict the high-impact damage percentage using the high-impact 
  ### SCM models (for high impact cases)
  ## wind and rainfall predictions are based on high impact data (damage >= 10)
  wind_max_pred_high <- predict(scm_models_high$wind_model_high, 
                         newdata = df)
  
  rain_total_pred_high <- predict(scm_models_high$rain_model_high, 
                         newdata = df)
  # add the predictions of wind and rainfall to the dataframe df
  df <- df %>%
    mutate(wind_max_pred = wind_max_pred_high, 
           rain_total_pred = rain_total_pred_high
           )
  
  high_pred <- predict(scm_models_high$high_reg_model, df)
  
  # Step 4: Apply the hurdle method logic
  predicted_damage <- ifelse(class_pred == 1, high_pred, base_pred)
  
  # Return the predicted damage
  return(predicted_damage)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predicting on base test set data
## because we already implemented the hurdle method 
df_test <- bind_rows(
  base_test,
  truncated_test
)
preds <- predictDamage(
  df = df_test,
  class_model = base_clas_min_model,
  scm_models_base = base_models,
  scm_models_high = high_models
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define bin edges
# Define bin edges
bins <- c(0, 0.00009, 1, 10, 50, 100)

# Assign data to bins
bin_labels <- cut(df_test$damage_perc, breaks = bins, include.lowest = TRUE, right = TRUE)

# Create a data frame with actual, predicted, and bin labels
data <- data.frame(
  actual = df_test$damage_perc,
  predicted = preds,
  bin = bin_labels
)

# Calculate RMSE per bin
unique_bins <- levels(data$bin) # Get unique bin labels
rmse_by_bin <- data.frame(bin = unique_bins, rmse = NA, count = NA) # Initialize results data frame

for (i in seq_along(unique_bins)) {
  bin_data <- data[data$bin == unique_bins[i], ] # Filter data for the current bin
  rmse_by_bin$rmse[i] <- sqrt(mean((bin_data$actual - bin_data$predicted)^2, na.rm = TRUE)) # Calculate RMSE
  rmse_by_bin$count[i] <- nrow(bin_data) # Count observations in the bin
}

# Display RMSE by bin
print(rmse_by_bin)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Writing binned RMSE to folder

# Recipe outputs
folder_path <- dkuManagedFolderPath("5NPBmWH1")

# Saving the predicted values
# Define file path
file_path <- file.path(folder_path, "rmse_by_bin.csv")

# Write to CSV
fwrite(as.data.frame(rmse_by_bin), file = file_path, row.names = FALSE)