library(dataiku)
library(dplyr)

# Recipe inputs
base_inc_data <- dkuReadDataset("base_inc_data", samplingMethod="head", nbRows=100000)

# Compute recipe outputs from inputs
base_modeling_data <- base_inc_data %>%



# Recipe outputs
dkuWriteDataset(base_modeling_data,"base_modeling_data")
