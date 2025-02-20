library(dataiku)

# Recipe inputs
modeling_data <- dkuReadDataset("modeling_data", samplingMethod="head", nbRows=100000)

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a R dataframe or data table
base_train <- modeling_data # For this sample code, simply copy input to output


# Recipe outputs
dkuWriteDataset(base_train,"base_train")
