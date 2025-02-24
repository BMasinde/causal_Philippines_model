# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
scm_base_models <- dkuManagedFolderPath("scm_base_models")
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# Reading .rds files
# Construct the full file path
wind_path <- file.path(scm_base_models, "base_wind_max_model.rds")

rain_path  <- file.path(scm_base_models, "base_rain_total_model.rds")

damage_path  <- file.path(scm_base_models, "damage_rain_total_model.rds")



# Read the RDS file
model <- readRDS(model_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
base_val_scm_models <- dkuManagedFolderPath("ZMqSlX9h")