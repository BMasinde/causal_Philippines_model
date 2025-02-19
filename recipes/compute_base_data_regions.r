library(dataiku)

# Recipe inputs
base_inc_data <- dkuReadDataset("base_inc_data", samplingMethod="head", nbRows=100000)

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a R dataframe or data table
base_data_regions <- base_inc_data # For this sample code, simply copy input to output


# Recipe outputs
dkuWriteDataset(base_data_regions,"base_data_regions")
