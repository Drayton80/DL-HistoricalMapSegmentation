import os
import glob
import argparse

import preprocessor
import trainer
import augmentor
import tester

ap = argparse.ArgumentParser()
ap.add_argument("-ag", "--augment", required=False, nargs='?', const=False, help="if the data will be augmented or not")
ap.add_argument("-p", "--preprocess", required=False, nargs='?', const=False, help="if the data will be preprocessed or not")
ap.add_argument("-tr", "--train", required=False, nargs='?', const=False, help="if the model will train or not")
ap.add_argument("-te", "--test", required=False, nargs='?', const=False, help="if the model will run the test validation or not")
ap.add_argument("-a", "--all", required=False, nargs='?', const=False, help="if will run all above")

args = vars(ap.parse_args())

if(args["all"] or args["augment"]):
    augmentor.run()
if(args["all"] or args["preprocess"]):
    preprocessor.run(compress_in='dataset_train.npz')
if(args["all"] or args["train"]):
    trainer.run('dataset_train.npz')
if(args["all"] or args["test"]):
    latest_model:str = max(glob.glob('./trained models/*.h5'), key=os.path.getctime)
    tester.run('trained models/' + latest_model)

