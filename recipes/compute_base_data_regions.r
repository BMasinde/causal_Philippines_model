# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adding Regions to the base_inc_data
library(dataiku)


# Regional data, including corresponding codes and names, are obtained from the  
# Philippine Statistics Authority website: https://psa.gov.ph/classification/psgc.  
# To access historical data, we use the Wayback Machine to retrieve a snapshot  
# from June 19, 2020. Specifically, we refer to the file named  
# *PSGC Publication March 2020.xlsx*. Since the file is small, we enter the data manually.
# We get Island groups from Wikipedia: https://en.wikipedia.org/wiki/Island_groups_of_the_Philippines
# NOTE: WE DO NOT HAVE THE NIR REGION IN THE DATASET

# creating a dataframe with regions:
region <- c('National Capital Region', 
            'Cordillera Administrative Region',
            'Ilocos Region', 
            'Cagayan Valley', 
            'Central Luzon',
            'CALABARZON', 
            'MIMAROPA Region',
            'Bicol Region',
            'Western Visayas',
            'Central Visayas',
            'Eastern Visayas',
            'Zamboanga Peninsula',
            'Northern Mindanao', 
            'Davao Region',
            'SOCCSKSARGEN',
            'Caraga',
            'ARMM')

region_code <- c('1300000000',
                 '1400000000', 
                 '0100000000',
                 '0200000000',
                 '0300000000',
                 '0400000000',
                 '1700000000',
                 '0500000000',
                 '0600000000',
                 '0700000000',
                 '0800000000',
                 '0900000000',
                 '1000000000',
                 '1100000000',
                 '1200000000',
                 '1600000000',
                 '1500000000')


island_groups <- c("Luzon", 
                   "Luzon", 
                   "Luzon", 
                   "Luzon", 
                   "Luzon", 
                   "Luzon", 
                   "Luzon", 
                   "Luzon",
                   "Visayas", 
                   "Visayas", 
                   "Visayas", 
                   "Mindanao", 
                   "Mindanao", 
                   "Mindanao",
                   "Mindanao", 
                   "Mindanao", 
                   "Mindanao"
                   )
# creating dataframe with the region names, regional codes, and respective island groups
regions_df <- as.data.frame(
    cbind(region_code, 
          region, island_groups
         )
)



# Recipe inputs
base_inc_data <- dkuReadDataset("base_inc_data", samplingMethod="head", nbRows=100000)

#
base_data_regions <- base_inc_data # For this sample code, simply copy input to output

#



# Recipe outputs
dkuWriteDataset(base_data_regions,"base_data_regions")