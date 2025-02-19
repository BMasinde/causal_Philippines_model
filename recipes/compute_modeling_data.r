library(dataiku)

# Recipe inputs
base_data_regions <- dkuReadDataset("base_data_regions", samplingMethod="head", nbRows=100000)

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a R dataframe or data table
modeling_data <- base_data_regions # For this sample code, simply copy input to output


# Recipe outputs
dkuWriteDataset(modeling_data,"modeling_data")
