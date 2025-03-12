library(dataiku)

# Recipe inputs
counterfactual_test_data <- dkuReadDataset("counterfactual_test_data", samplingMethod="head", nbRows=100000)
hurdle_predictions_testing <- dkuManagedFolderPath("5NPBmWH1")



# Recipe outputs
matching_counterfactuals <- dkuManagedFolderPath("ZO3oPxC1")
