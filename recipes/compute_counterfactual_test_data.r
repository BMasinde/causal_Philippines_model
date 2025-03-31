library(dataiku)

# Recipe inputs
base_data_regions <- dkuReadDataset("base_data_regions", samplingMethod="head", nbRows=100000)
hurdle_predictions_testing <- dkuManagedFolderPath("5NPBmWH1")

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a R dataframe or data table
counterfactual_test_data <- base_data_regions # For this sample code, simply copy input to output


# Recipe outputs
dkuWriteDataset(counterfactual_test_data,"counterfactual_test_data")
