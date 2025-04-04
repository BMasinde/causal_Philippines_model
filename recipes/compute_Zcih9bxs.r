library(dataiku)

# Recipe inputs
hurdle_predictions_testing <- dkuManagedFolderPath("5NPBmWH1")
counterfactual_test_data <- dkuReadDataset("counterfactual_test_data", samplingMethod="head", nbRows=100000)



# Recipe outputs
fixed_sec_hazards_counterfactuals <- dkuManagedFolderPath("Zcih9bxs")
