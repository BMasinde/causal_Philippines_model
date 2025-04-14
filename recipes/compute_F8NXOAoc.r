library(dataiku)

# Recipe inputs
base_data_regions <- dkuReadDataset("base_data_regions", samplingMethod="head", nbRows=100000)



# Recipe outputs
v510_counterfactual_datasets_fixed_sec_hazards <- dkuManagedFolderPath("F8NXOAoc")
