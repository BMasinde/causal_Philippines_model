# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)
library(dplyr)
library(caret)
library(data.table)
library(mlflow)
library(reticulate)
library(Metrics)
library(themis)
library(doMC)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Combining train and validation datasets to one
# Because we are going to use CV to train the models later
# naming it df_base_train2 to remain consistent with df naming
df_base_train2  <- rbind(base_train, base_validation)

cat("number of rows in combined train data:", nrow(df_base_train2), sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# setting seed for reproducibility
set.seed(1234)

# Define tuning grid
tune_grid <- expand.grid(
  nrounds = c(47,50, 60,70), # early stopping does not work, we still need to specify nrounds
  max_depth = c(2, 3, 4, 6),
  eta = c(0.09, 0.1, 0.11, 0.12),
  gamma = c(0, 1, 2, 3, 4),
  colsample_bytree = c(0.9, 1.0, 1.1),
  min_child_weight = c(2, 3, 4),
  subsample = c(0.5, 0.6, 0.7, 0.8)
)


# Set up train control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 7,
  summaryFunction = defaultSummary,
  search = "random" # random selection of the expanded grid
)

# Detect and register the number of available cores (use all but one)
num_cores <- parallel::detectCores() - 2
registerDoMC(cores = num_cores)  # Enable parallel processing

# Measure the time for a code block to run
system.time({
# Train the model using grid search with 7-fold CV
base_xgb_reg_model <- train(
  damage_perc ~ track_min_dist +
    wind_max +
    rain_total +
    roof_strong_wall_strong +
    roof_strong_wall_light +
    roof_strong_wall_salv +
    roof_light_wall_strong +
    roof_light_wall_light +
    roof_light_wall_salv +
    roof_salv_wall_strong +
    roof_salv_wall_light +
    roof_salv_wall_salv +
    blue_ss_frac +
    yellow_ss_frac +
    orange_ss_frac +
    red_ss_frac +
    blue_ls_frac +
    yellow_ls_frac +
    orange_ls_frac +
    red_ls_frac,
  data = df_base_train2,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE"  # Optimize based on RMSE
)
Sys.sleep(2)  # This is just an example to simulate a delay
})
# Print best parameters
print(base_xgb_reg_model$bestTune)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------- best parameters ---------------
best_params <- base_xgb_reg_model$bestTune

# ---------- train using best parameters
damage_fit_reg_base <- train(damage_perc ~ track_min_dist +
                               wind_max +
                               rain_total +
                               roof_strong_wall_strong +
                               roof_strong_wall_light +
                               roof_strong_wall_salv +
                               roof_light_wall_strong +
                               roof_light_wall_light +
                               roof_light_wall_salv +
                               roof_salv_wall_strong +
                               roof_salv_wall_light +
                               roof_salv_wall_salv +
                               blue_ss_frac +
                               yellow_ss_frac +
                               orange_ss_frac +
                               red_ss_frac +
                               blue_ls_frac +
                               yellow_ls_frac +
                               orange_ls_frac +
                               red_ls_frac,
                              method = "xgbTree",
                              trControl = trainControl(method = "none"),
                              tuneGrid = best_params, # Use the best parameters here
                              metric = "RMSE",
                              data = df_base_train2
                         )

# obtain predicted values
train_predictions <- predict(damage_fit_reg_base, newdata = df_base_train2)


# Define bin edges
# Define bin edges
bins <- c(0.00009, 1, 10, 50, 100)

# Assign data to bins
bin_labels <- cut(df_base_train2$damage_perc, breaks = bins, include.lowest = TRUE, right = TRUE)

# Create a data frame with actual, predicted, and bin labels
data <- data.frame(
  actual = df_base_train2$damage_perc,
  predicted = train_predictions,
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

as.data.frame(rmse_by_bin)
RMSE_1 <- rmse_by_bin[1, "rmse"]
RMSE_10 <-  rmse_by_bin[2, "rmse"]
RMSE_50 <- rmse_by_bin[3, "rmse"]
RMSE_100 <- rmse_by_bin[4, "rmse"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Testing the model on out of sample dataset
# obtain predicted values
test_predictions <- predict(damage_fit_reg_base, newdata = base_test)

# Define bin edges for test
bins <- c(0.00009, 1, 10, 50, 100)
# Assign data to bins
test_bin_labels <- cut(base_test$damage_perc, breaks = bins, include.lowest = TRUE, right = TRUE)
# Create a data frame with actual, predicted, and bin labels
test_data <- data.frame(
  actual = base_test$damage_perc,
  predicted = test_predictions,
  bin = test_bin_labels
)

# Calculate RMSE per bin
unique_bins <- levels(test_data$bin) # Get unique bin labels
rmse_by_bin <- data.frame(bin = unique_bins, rmse = NA, count = NA) # Initialize results data frame

for (i in seq_along(unique_bins)) {
  test_bin_data <- test_data[test_data$bin == unique_bins[i], ] # Filter data for the current bin
  rmse_by_bin$rmse[i] <- sqrt(mean((test_bin_data$actual - test_bin_data$predicted)^2, na.rm = TRUE)) # Calculate RMSE
  rmse_by_bin$count[i] <- nrow(test_bin_data) # Count observations in the bin
}

# Display RMSE by bin
print(rmse_by_bin)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
ass_XGBOOST_base_reg <- dkuManagedFolderPath("DFdq9elg")

# save the trained truncated model
model_filename <- "damage_fit_reg_base.rds"

saveRDS(damage_fit_reg_base, file = file.path(ass_XGBOOST_base_reg, model_filename))