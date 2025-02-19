# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adding Regions to the base_inc_data
library(dataiku)
library(dplyr)

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
# Creating dataframe with the region names, regional codes, and respective island groups
regions_df <- as.data.frame(
    cbind(region_code,
          region, island_groups
         )
)


# Recipe inputs
base_inc_data <- dkuReadDataset("base_inc_data", samplingMethod="head", nbRows=100000)

# Creating base_data_regions which will be the output of this recipe
base_data_regions <- base_inc_data # For this sample code, simply copy input to output

# Correct length of Mun_Code_2 (should be 9 or 10 digits)
base_data_regions <- within(base_data_regions, {
  Mun_Code_2 <- as.character(Mun_Code_2)  # Ensure it's a character type
  
  Mun_Code_2 <- ifelse(
    nchar(Mun_Code_2) < 9, 
    paste0("0", Mun_Code_2),  # Append zero if length is less than 9
    Mun_Code_2                # Otherwise, keep as is
  )
})

# Joining base_data_regions with regions_df to add the region columns
base_data_regions <- base_data_regions %>%
  # Extract the first two characters of D
  mutate(Key = substr(Mun_Code_2, 1, 2)) %>%
  # Perform a left join with B
  left_join(
    regions_df %>%
      mutate(Key = substr(region_code, 1, 2)) %>% 
      select(Key, region, island_groups), # Keep only the relevant columns from B
    by = "Key"
  ) %>%
  select(-Key) # Drop the intermediate Key column


# Recipe outputs
dkuWriteDataset(base_data_regions,"base_data_regions")