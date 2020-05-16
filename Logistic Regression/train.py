# Logistic Regression

# importing the required libraries
import time
import argparse
import os
import sys
if sys.version_info >= (3, 0):
        import _pickle as cPickle
else:
        import cPickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS
from confusion_matrix import func_confusion_matrix


def train(max_iter=HYPERPARAMS.max_iter, train_model=True):

        print( "Loading the dataset: " + DATASET.name)
        if train_model:
                data, validation = load_data(validation=True)
        else:
                data, validation, test = load_data(validation=True, test=True)
        
        if train_model:
            # training phase
            print( "Building the model!")
            model = LogisticRegression(max_iter=max_iter)

            print( "Starting the training...")
            print( "--")
            print("max_iter: {}".format(max_iter))
            print( "--")
            print( "Training samples: {}".format(len(data['Y'])))
            print( "Validation samples: {}".format(len(validation['Y'])))
            print( "--")
            
            start_time = time.time()
            model.fit(data['X'], data['Y'])
            training_time = time.time() - start_time
            print( "Training time: {0:.1f} sec".format(training_time))

            if TRAINING.save_model:
                print( "Saving the model.")
                with open(TRAINING.save_model_path, 'wb') as f:
                        cPickle.dump(model, f)

            print( "Evaluating the model!")
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print( "Validation accuracy: {0:.1f}".format(validation_accuracy*100))
            return validation_accuracy
        else:
            # Testing phase : load saved model and evaluate on test dataset
            print( "Starting the evaluation of the model...")
            print( "Loading the pretrained model.")
            if os.path.isfile(TRAINING.save_model_path):
                with open(TRAINING.save_model_path, 'rb') as f:
                        model = cPickle.load(f)
            else:
                print( "Error: File '{}' not found.".format(TRAINING.save_model_path))
                exit()

            print( "--")
            print( "Validation samples: {}".format(len(validation['Y'])))
            print( "Test samples: {}".format(len(test['Y'])))
            print( "--")
            print( "Evaluating the model!")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'],  validation['Y'])
            print( "Validation accuracy: {0:.1f}".format(validation_accuracy*100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print( "Test accuracy: {0:.1f}".format(test_accuracy*100))
            print( "Evalution time: {0:.1f} sec".format(time.time() - start_time))
            return test_accuracy

def evaluate(model, X, Y):
        predicted_Y = model.predict(X)
        accuracy = accuracy_score(Y, predicted_Y)
        return accuracy

def evaluate_confusion(model, X, Y):
        predicted_Y = model.predict(X)
        conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(Y, predicted_Y)
        return accuracy, conf_matrix

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
        train(train_model=False)