import os
import glob
import argparse

import preprocess
import training
import augment
import test

ap = argparse.ArgumentParser()
ap.add_argument("-ag", "--augment", required=False, nargs='?', const=True, help="if the data will be augmented or not")
ap.add_argument("-p", "--preprocess", required=False, nargs='?', const=True, help="if the data will be preprocessed or not")
ap.add_argument("-tr", "--training", required=False, nargs='?', const=True, help="if the model will be training or not")
ap.add_argument("-te", "--test", required=False, nargs='?', const=True, help="if the model will run the test validation or not")
ap.add_argument("-a", "--all", required=False, nargs='?', const=True, help="if will run all above")

args = vars(ap.parse_args())

if(args["all"] or args["augment"]):
    augment.run()
if(args["all"] or args["preprocess"]):
    preprocess.run()
if(args["all"] or args["training"]):
    training.run()
if(args["all"] or args["test"]):
    latest_model:str = max(glob.glob('./trained models/*.h5'), key=os.path.getctime).replace('\\', '/')
    test.run(latest_model)

