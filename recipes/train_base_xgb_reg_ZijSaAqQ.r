# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Libraries
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
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Combining train and validation datasets to one
# Because we are going to use CV to train the models later
# naming it df_base_train2 to remain consistent with df naming
df_base_train2  <- rbind(df_base_train, df_base_validation)

cat("number of rows in combined train data:", nrow(df_base_train2), sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training track_min_dist ~ island_groups
# we will need to also include island_groups
# in the final outcome prediction model to adjust for the confounding

base_track_model  <- rpart(track_min_dist  ~ island_groups,
                          data = df_base_train2,
                          method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for wind speed
# wind_speed = f(track_min_dist, eps)

base_wind_model <- rpart(wind_max ~ track_min_dist,
                       data = df_base_train2,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for rain speed
# rain_total = f(track_min_dist, eps)

base_rain_model <- rpart(rain_total ~ track_min_dist,
                       data = df_base_train2,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Building typologies are determined by region
base_roof_strong_wall_strong_model  <- rpart(roof_strong_wall_strong  ~ island_groups, 
                                             data = df_base_train2,
                                            method = "anova")

base_roof_strong_wall_light_model  <- rpart(roof_strong_wall_light ~ island_groups,
                                           data = df_base_train2,
                                           method = "anova")

base_roof_strong_wall_salv_model  <- rpart(roof_strong_wall_salv ~ island_groups,
                                          data = df_base_train2,
                                          method = "anova")
base_roof_light_wall_strong_model  <- rpart(roof_light_wall_strong ~ island_groups,
                                           data = df_base_train2,
                                           method = "anova")
base_roof_light_wall_light_model  <- rpart(roof_light_wall_light ~ island_groups,
                                          data = df_base_train2,
                                          method = "anova")
base_roof_light_wall_salv_model  <- rpart(roof_light_wall_salv ~ island_groups,
                                         data = df_base_train2,
                                         method = "anova")

base_roof_salv_wall_strong_model  <- rpart(roof_salv_wall_strong ~ island_groups,
                                          data = df_base_train2,
                                          method = "anova")

base_roof_salv_wall_light_model  <- rpart(roof_salv_wall_light ~ island_groups,
                                  data = df_base_train2,
                                  method = "anova")

base_roof_salv_wall_salv_model  <- rpart(roof_salv_wall_salv ~ island_groups,
                                  data = df_base_train2,
                                  method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # Adding the predicted parents' to the training dataset

# df_base_train <- base_train %>%
#   mutate(wind_max_pred = predict(base_wind_model,
#                                  newdata = df_base_train2),
#          rain_total_pred = predict(base_rain_model,
#                                    newdata = base_train)
#          )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_list <- list(
  track_min_dist = base_track_model,
  wind_max = base_wind_model,
  rain_total = base_rain_model,
  roof_strong_wall_strong = base_roof_strong_wall_strong_model,
  roof_strong_wall_light = base_roof_strong_wall_light_model,
  roof_strong_wall_salv = base_roof_strong_wall_salv_model,
  roof_light_wall_strong = base_roof_light_wall_strong_model,
  roof_light_wall_light = base_roof_light_wall_light_model,
  roof_light_wall_salv = base_roof_light_wall_salv_model,
  roof_salv_wall_strong = base_roof_salv_wall_strong_model,
  roof_salv_wall_light = base_roof_salv_wall_light_model,
  roof_salv_wall_salv = base_roof_salv_wall_salv_model
)

# Apply predictions efficiently
df_base_train2 <- df_base_train2 %>%
  mutate(across(names(model_list), ~ predict(model_list[[cur_column()]], newdata = df_base_train2), .names = "{.col}_pred")) 

# Define wind and rain interaction variables
wind_fractions <- c("blue_ss_frac", "yellow_ss_frac", "orange_ss_frac", "red_ss_frac")
rain_fractions <- c("blue_ls_frac", "yellow_ls_frac", "orange_ls_frac", "red_ls_frac")

# Compute wind interaction terms dynamically
df_base_train2 <- df_base_train2 %>%
  mutate(across(all_of(wind_fractions), ~ . * wind_max_pred, .names = "wind_{.col}"),
         across(all_of(rain_fractions), ~ . * rain_total_pred, .names = "rain_{.col}"))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # parameter tuning
# # Define a grid of hyperparameters
# cp_values <- seq(0.0001, 0.05, by = 0.0005)
# maxdepth_values <- c(3, 5, 7, 10)
# minsplit_values <- c(10, 20, 30, 40)
# minbucket_values <- c(5, 10, 20)

# # Create an empty list to store results
# # Create an empty list to store results
# results <- data.frame(cp = numeric(), maxdepth = numeric(),
#                       minsplit = numeric(), minbucket = numeric(), RMSE = numeric())

# # predicting for wind and rainfall for the validation dataset
# df_val_base_tune <- base_validation %>%
#   mutate(
#     wind_max_pred = predict(
#       base_wind_model, newdata = base_validation),
#     rain_total_pred = predict(
#       base_rain_model,
#       newdata = base_validation)
#     )

# # Train the model using manual grid search
# grid_id <- 1  # Index for list storage

# # Iterate over all combinations of hyperparameters
# for (cp in cp_values) {
#   for (maxdepth in maxdepth_values) {
#     for (minsplit in minsplit_values) {
#       for (minbucket in minbucket_values) {

#         # Train the model with specific hyperparameters
#         model <- rpart(
#           damage_perc ~ wind_max_pred +
#             rain_total_pred +
#             roof_strong_wall_strong +
#             roof_strong_wall_light +
#             roof_strong_wall_salv +
#             roof_light_wall_strong +
#             roof_light_wall_light +
#             roof_light_wall_salv +
#             roof_salv_wall_strong +
#             roof_salv_wall_light +
#             roof_salv_wall_salv +
#             ls_risk_pct +
#             ss_risk_pct +
#             wind_blue_ss +
#             wind_yellow_ss +
#             wind_orange_ss +
#             wind_red_ss +
#             rain_blue_ss +
#             rain_yellow_ss +
#             rain_orange_ss +
#             rain_red_ss,
#           data = df_base_train,
#           method = "anova",  # Regression
#           control = rpart.control(cp = cp, maxdepth = maxdepth,
#                                   minsplit = minsplit, minbucket = minbucket)
#         )

#         # Make predictions on the validation set
#         val_predictions <- predict(model, newdata = df_val_base_tune)

#         # Compute RMSE
#         rmse_value <- rmse(df_val_base_tune$damage_perc, val_predictions)

#         # Store results
#         results <- rbind(results, data.frame(cp, maxdepth, minsplit, minbucket, RMSE = rmse_value))
#       }
#     }
#   }
# }

# # Print the best hyperparameter combination
# best_params <- results[which.min(results$RMSE), ]
# print(best_params)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define tuning grid
tune_grid <- expand.grid(
  nrounds = c(50, 100, 200, 300, 400, 500),
  max_depth = c(3, 6, 9, 12),
  eta = c(0.01, 0.05, 0.1, 0.2, 0.3),
  gamma = c(0, 1, 5, 10),
  colsample_bytree = c(0.5, 0.7, 0.8, 1.0),
  min_child_weight = c(1, 3, 5, 10),
  subsample = c(0.5, 0.7, 0.8, 1.0)
)


# Set up train control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,  # Needed for AUC calculation
  summaryFunction = twoClassSummary
)

# Train the model using grid search with 10-fold CV
set.seed(1234)
base_xgb_reg_model <- train(
  damage_pred ~ wind_max_pred +
    rain_total_pred +
    roof_strong_wall_strong_pred +
    roof_strong_wall_light_pred +
    roof_strong_wall_salv_pred +
    roof_light_wall_strong_pred +
    roof_light_wall_light_pred +
    roof_light_wall_salv_pred +
    roof_salv_wall_strong_pred +
    roof_salv_wall_light_pred +
    roof_salv_wall_salv_pred +
    ls_risk_pct +
    ss_risk_pct +
    wind_blue_ss +
    wind_yellow_ss +
    wind_orange_ss +
    wind_red_ss +
    rain_blue_ss +
    rain_yellow_ss +
    rain_orange_ss +
    rain_red_ss +
    island_groups +  # Confounder adjustment
    track_min_dist_pred, # Confounder adjustment
  data = df_base_train2,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE"  # Optimize based on AUC
)

# Print best parameters
print(base_xgb_model$bestTune)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # Training based on tuned parameters

# # Combine Training and Validation datasets for final training

# final_training_df  <- rbind(df_base_train,
#                            df_val_base_tune)


# damage_fit_reg_min <- rpart(damage_perc ~ wind_max_pred +
#                               rain_total_pred +
#                               roof_strong_wall_strong +
#                               roof_strong_wall_light +
#                               roof_strong_wall_salv +
#                               roof_light_wall_strong +
#                               roof_light_wall_light +
#                               roof_light_wall_salv +
#                               roof_salv_wall_strong +
#                               roof_salv_wall_light +
#                               roof_salv_wall_salv +
#                               ls_risk_pct +
#                               ss_risk_pct +
#                               wind_blue_ss +
#                               wind_yellow_ss +
#                               wind_orange_ss +
#                               wind_red_ss +
#                               rain_blue_ss +
#                               rain_yellow_ss +
#                               rain_orange_ss +
#                               rain_red_ss,
#                               method = "anova",
#                               control = rpart.control(cp = best_params$cp,
#                                                       maxdepth = best_params$maxdepth,
#                                                       minsplit = best_params$minsplit,
#                                                       minbucket = best_params$minbucket),
#                               data = final_training_df
#                          )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ----------------------       Logging model performanc using MLFLOW ------------------

# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "SCM - XGBOOST base regression - CV (Training metircs)")

# Ensure that MLflow has only one run. Start MLflow run once.
run_name <- paste("XGBoost Run", Sys.time())  # Unique name using current time


# Start MLflow run
mlflow_start_run(nested = FALSE)

# Ensure the run ends even if an error occurs
#on.exit(mlflow_end_run(), add = TRUE)


# -------- best parameters ---------------
best_params <- base_xgb_reg_model$bestTune

# Log each of the best parameters in MLflow
for (param in names(best_params)) {
  mlflow_log_param(param, best_params[[param]])
}

# ---------- train using best parameters
damage_fit_reg_min <- train(damage_perc ~ wind_max_pred +
                              rain_total_pred +
                              roof_strong_wall_strong_pred +
                              roof_strong_wall_light_pred +
                              roof_strong_wall_salv_pred +
                              roof_light_wall_strong_pred +
                              roof_light_wall_light_pred +
                              roof_light_wall_salv_pred +
                              roof_salv_wall_strong_pred +
                              roof_salv_wall_light_pred +
                              roof_salv_wall_salv_pred +
                              ls_risk_pct +
                              ss_risk_pct +
                              wind_blue_ss +
                              wind_yellow_ss +
                              wind_orange_ss +
                              wind_red_ss +
                              rain_blue_ss +
                              rain_yellow_ss +
                              rain_orange_ss +
                              rain_red_ss +
                              island_groups +  # Confounder adjustment
                              track_min_dist_pred, # Confounder adjustment
                              method = "xgbTree",
                              trControl = trainControl(method = "none"),
                              tuneGrid = best_params, # Use the best parameters here
                              metric = "RMSE" 
                              data = df_base_train2
                         )

# obtain predicted values
train_predictions <- damage_fit_reg_min$pred$pred


# Define bin edges
# Define bin edges
bins <- c(0.00009, 1, 10, 50, 100)

# Assign data to bins
bin_labels <- cut(df_base_train2$damage_perc, breaks = bins, include.lowest = TRUE, right = TRUE)

# Create a data frame with actual, predicted, and bin labels
data <- data.frame(
  actual = df_test$damage_perc,
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

# Log binned RMSE metrics  
mlflow_log_metric("RMSE \[0.00009, 1]", rmse_by_bin[1, 1])
mlflow_log_metric("RMSE \[1, 10]", rmse_by_bin[2, 1])
mlflow_log_metric("RMSE \[10, 50]", rmse_by_bin[3, 1])
mlflow_log_metric("RMSE \[50, 100]", rmse_by_bin[4, 1])

# End MLflow run
mlflow_end_run()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Sanity Check
# RMSE on the trainset (training + validation)
# Compute RMSE

damage_pred  <- predict(damage_fit_reg_min, newdata = df_base_train2)
rmse_value <- rmse(final_training_df$damage_perc, damage_pred)
rmse_value

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
managed_folder_path <- dkuManagedFolderPath("ZijSaAqQ")

# ------------ Models in a list -----------------------
models <- list(damage_fit_reg_min,
               base_wind_model, 
               base_rain_model, 
               base_track_model,
               base_roof_strong_wall_strong_model,
               base_roof_strong_wall_light_model,
               base_roof_strong_wall_salv_model,
               base_roof_light_wall_strong_model,
               base_roof_light_wall_light_model,
               base_roof_light_wall_salv_model, 
               base_roof_salv_wall_strong_model,
               base_roof_salv_wall_light_model,
               base_roof_salv_wall_salv_model
              )
model_names <- c("base_reg_min_model",
                 "base_wind_model", 
                 "base_rain_model", 
                 "base_track_model",
                 "base_roof_strong_wall_strong_model",
                 "base_roof_strong_wall_light_model",
                 "base_roof_strong_wall_salv_model",
                 "base_roof_light_wall_strong_model",
                 "base_roof_light_wall_light_model",
                 "base_roof_light_wall_salv_model", 
                 "base_roof_salv_wall_strong_model",
                 "base_roof_salv_wall_light_model",
                 "base_roof_salv_wall_salv_model"
                )

#----------------------- Saving trained XGBOOST model ----------------------------------------
mapply(function(model, name) {
  saveRDS(model, file = paste0(managed_folder_path, "/", name, ".rds"))
}, models, model_names)


# saveRDS(damage_fit_reg_min, file = paste0(managed_folder_path, "/base_reg_min_model.rds"))

# saveRDS(base_wind_model, file = paste0(managed_folder_path, "/base_wind_model.rds"))

# saveRDS(base_rain_model, file = paste0(managed_folder_path, "/base_rain_model.rds"))