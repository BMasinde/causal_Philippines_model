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
# Save the trained model in a Managed Folder
dkuManagedFolderPath <- dkuManagedFolderPath("scm_models")
saveRDS(base_wind_max_model, file = paste0(dkuManagedFolderPath, "/base_wind_max_model.rds"))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
scm_models <- dkuManagedFolderPath("XxDUuuYe")