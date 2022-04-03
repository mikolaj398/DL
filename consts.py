MAIN_DATASET_IMGS_PATH = 'main_dataset'
COVID_DATASET_IMGS_PATH = 'covid_dataset'

MAIN_DATASET_ANNOTATIONS_PATH = 'main_labels.csv'
COVID_DATASET_ANNOTATIONS_PATH = 'covid_labels.csv'

IMG_SIZE = (128, 128)

columns_mapping = {
    'filename': 'Image Index',
    'patientid': 'Patient ID',
    'finding': 'Finding Labels',
    'age': 'Patient Age',
    'sex': 'Patient Gender',
    'path': 'path',

}