# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the dataset as a R dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
df_base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for wind speed
# wind_speed = f(track_min_dist, eps)

base_wind_max_model <- rpart(wind_max ~ track_min_dist,
                       data = df_base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for rain speed
# rain_total = f(track_min_dist, eps)

base_rain_total_model <- rpart(rain_total ~ track_min_dist,
                       data = df_base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## Adding the predicted parents' to the training dataset
df_base_train <- df_base_train %>%
  mutate(wind_max_pred = wind_max_pred, 
         rain_total_pred = rain_total_pred
         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training decision tree for classification 
damage_fit_class <- rpart(Y ~ wind_max_pred + 
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
                           ruggedness_mean +
                           slope_mean +
                           wind_blue_ss +
                           wind_yellow_ss +          
                           wind_orange_ss +          
                           wind_red_ss +
                           rain_blue_ss +
                           rain_yellow_ss +
                           rain_orange_ss +
                           rain_red_ss,
                         method = "class", 
                         data = df_base_train
                         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training decision tree for regression
base_damage_fit_reg <- rpart(damage_perc ~ wind_max_pred + 
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
                           ruggedness_mean +
                           slope_mean +
                           wind_blue_ss +
                           wind_yellow_ss +          
                           wind_orange_ss +          
                           wind_red_ss +
                           rain_blue_ss +
                           rain_yellow_ss +
                           rain_orange_ss +
                           rain_red_ss,
                         method = "class", 
                         data = df_base_train
                         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save the trained models in a Managed Folder
dkuManagedFolderPath <- dkuManagedFolderPath("scm_base_models")

# Saving wind model
saveRDS(base_wind_max_model, file = paste0(dkuManagedFolderPath, "/base_wind_max_model.rds"))

# Saving rain model
saveRDS(base_rain_total_model, file = paste0(dkuManagedFolderPath, "/base_rain_total_model.rds"))

# Saving classification model
saveRDS(damage_fit_class, file = paste0(dkuManagedFolderPath, "/damage_fit_class_model.rds"))

# Saving base regression model
saveRDS(base_damage_fit_reg, file = paste0(dkuManagedFolderPath, "/base_damage_fit_reg_model.rds"))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
colnames(df_base_train)