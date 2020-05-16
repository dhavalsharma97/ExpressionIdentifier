# importing the required libraries
import os


class Dataset:
    name = 'Fer2013'
    train_folder = 'features/Training'
    validation_folder = 'features/PublicTest'
    test_folder = 'features/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1


class Hyperparams:
    features = "landmarks_and_hog" # "landmarks" or "hog" or "landmarks_and_hog" 
 
    
class Training:
    save_model = True
    save_model_path = "saved_model.bin"


DATASET = Dataset()
TRAINING = Training()
HYPERPARAMS = Hyperparams()