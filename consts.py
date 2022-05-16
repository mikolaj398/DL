MAIN_DATASET_IMGS_PATH = 'main_dataset'
COVID_DATASET_IMGS_PATH = 'covid_dataset'

MAIN_DATASET_ANNOTATIONS_PATH = 'main_labels.csv'
COVID_DATASET_ANNOTATIONS_PATH = 'covid_labels.csv'

PLOTS_PATH = './plots/'
RESULTS_PATH = './results/'

EPOCHS = 30
BATCH_SIZE = 24
STEPS_PER_EPOCH = 10

IMG_SIZE = (128, 128)

columns_mapping = {
    'filename': 'Image Index',
    'patientid': 'Patient ID',
    'finding': 'Finding Labels',
    'age': 'Patient Age',
    'sex': 'Patient Gender',
    'path': 'path',

}

