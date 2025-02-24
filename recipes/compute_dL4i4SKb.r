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
truncated_train <- dkuReadDataset("truncated_train", samplingMethod="head", nbRows=100000)
truncated_validation <- dkuReadDataset("truncated_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Fitting tree for wind and rain
# wind_max prediction using decision trees

trunc_wind_model <- rpart(wind_max ~ track_min_dist, 
                       data = truncated_train, 
                       method = "anova")

trunc_rain_model <- rpart(rain_total ~ track_min_dist, 
                       data = truncated_train, 
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# add the predictions of wind and rainfall to the dataframes
df_trunc_train <- truncated_train %>%
  mutate(wind_max_pred = predict(trunc_wind_model, 
                         newdata = truncated_train), 
         rain_total_pred = predict(trunc_rain_model, 
                         newdata = truncated_train)
        )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predicting for wind and rainfall for the validation dataset on trained high impact
# WE NEED THIS FOR HYPERPARAMETER TUNING!
df_trunc_val <- truncated_validation %>%
  mutate(
    wind_max_pred = predict(
      trunc_wind_model, newdata = truncated_validation),
    rain_total_pred = predict(
      trunc_rain_model, 
      newdata = truncated_validation)
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define a grid of hyperparameters same as used for the base model
cp_values <- seq(0.0001, 0.05, by = 0.0005)
maxdepth_values <- c(3, 5, 7, 10)
minsplit_values <- c(10, 20, 30, 40)
minbucket_values <- c(5, 10, 20)

# Create an empty list to store results
results <- data.frame(cp = numeric(), maxdepth = numeric(), 
                      minsplit = numeric(), minbucket = numeric(), RMSE = numeric())



# Train the model using manual grid search
# Iterate over all combinations of hyperparameters
for (cp in cp_values) {
  for (maxdepth in maxdepth_values) {
    for (minsplit in minsplit_values) {
      for (minbucket in minbucket_values) {
        
        # Train the model with specific hyperparameters
        model <- rpart(
          damage_perc ~ wind_max_pred + 
            rain_total_pred + 
            roof_strong_wall_strong + 
            roof_strong_wall_light + 
            roof_strong_wall_salv + 
            roof_light_wall_strong + 
            roof_light_wall_light + 
            roof_light_wall_salv + 
            roof_salv_wall_strong + 
            roof_salv_wall_light + 
            roof_salv_wall_salv + 
            ls_risk_pct + 
            ss_risk_pct + 
            wind_blue_ss + 
            wind_yellow_ss + 
            wind_orange_ss + 
            wind_red_ss + 
            rain_blue_ss + 
            rain_yellow_ss + 
            rain_orange_ss + 
            rain_red_ss,
          data = df_trunc_train,
          method = "anova",  # Regression tree
          control = rpart.control(cp = cp, maxdepth = maxdepth, 
                                  minsplit = minsplit, minbucket = minbucket)
        )
        
        # Make predictions on the validation set
        val_predictions <- predict(model, newdata = df_trunc_val)
        
        # Compute RMSE
        rmse_value <- rmse(df_trunc_val$damage_perc, val_predictions)
        
        # Store results
        results <- rbind(results, data.frame(cp, maxdepth, minsplit, minbucket, RMSE = rmse_value))
      }
    }
  }
}

# Print the best hyperparameter combination
best_params <- results[which.min(results$RMSE), ]
print(best_params)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training based on tuned parameters
# Combining training data and validation data
final_training_df  <- rbind(df_trunc_train,
                           df_trunc_val)

trunc_damage_fit_reg <- rpart(damage_perc ~ wind_max_pred + 
                           rain_total_pred + 
                           roof_strong_wall_strong + 
                           roof_strong_wall_light +
                           roof_strong_wall_salv + 
                           roof_light_wall_strong + 
                           roof_light_wall_light + 
                           roof_light_wall_salv + 
                           roof_salv_wall_strong +
                           roof_salv_wall_light +
                           roof_salv_wall_salv +
                           ls_risk_pct +
                           ss_risk_pct +
                           wind_blue_ss +
                           wind_yellow_ss +          
                           wind_orange_ss +          
                           wind_red_ss +
                           rain_blue_ss +
                           rain_yellow_ss +
                           rain_orange_ss +
                           rain_red_ss, 
                         method = "anova", 
                         data = final_training_df
                             )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Sanity Check
# RMSE on the trainset (training + validation)
# Compute RMSE

damage_pred  <- predict(trunc_damage_fit_reg, newdata = final_training_df)
rmse_value <- rmse(final_training_df$damage_perc, damage_pred)
rmse_value

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
managed_folder_path <- dkuManagedFolderPath("dL4i4SKb")

saveRDS(trunc_damage_fit_reg, file = paste0(managed_folder_path, "/trunc_reg_min_model.rds"))

saveRDS(trunc_wind_model, file = paste0(managed_folder_path, "/trunc_wind_model.rds"))

saveRDS(trunc_rain_model, file = paste0(managed_folder_path, "/trunc_rain_model.rds"))