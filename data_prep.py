from consts import *
import pandas as pd
from glob import glob
import os


def get_combined_datasets():
    main_df = pd.read_csv(MAIN_DATASET_ANNOTATIONS_PATH)
    covid_df = pd.read_csv(COVID_DATASET_ANNOTATIONS_PATH)
    covid_df.rename(columns=columns_mapping, inplace=True)

    main_df_image_paths = {os.path.basename(x): x for x in 
                            glob(os.path.join('.', MAIN_DATASET_IMGS_PATH, '*'))}
    
    covid_df_image_paths = {os.path.basename(x): x for x in 
                            glob(os.path.join('.', COVID_DATASET_IMGS_PATH, '*'))}

    main_df['path'] = main_df['Image Index'].map(main_df_image_paths.get)
    covid_df['path'] = covid_df['Image Index'].map(covid_df_image_paths.get)

    covid_df = covid_df[list(columns_mapping.values())]
    main_df = main_df[list(columns_mapping.values())]

    covid_df = get_covid_cases(covid_df)

    return pd.concat([main_df, covid_df])

def get_covid_cases(covid_df):
    covid_cases_df = covid_df[covid_df['Finding Labels'].str.contains('COVID-19')]
    covid_cases_df['Finding Labels'] = 'COVID-19'
    covid_cases_df.dropna(inplace=True)

    return covid_cases_df


