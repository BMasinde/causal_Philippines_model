library(dataiku)

# Recipe inputs
ass_hurdle_predictions <- dkuManagedFolderPath("K9arRZ6K")
counterfactual_test_data <- dkuReadDataset("counterfactual_test_data", samplingMethod="head", nbRows=100000)



# Recipe outputs
ass_fixed_sec_hazards_counterfactuals <- dkuManagedFolderPath("V3ui5Jrj")
