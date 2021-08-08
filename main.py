import argparse

from Preprocessor import Preprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-ag", "--augment", required=False, nargs='?', const=False, help="if the data will be augmented or not")
ap.add_argument("-p", "--preprocess", required=False, nargs='?', const=False, help="if the data will be preprocessed or not")
ap.add_argument("-tm", "--trainmodel", required=False, nargs='?', const=False, help="if the model will train or not")
ap.add_argument("-rm", "--runmodel", required=False, nargs='?', const=False, help="if the model will run or not")
ap.add_argument("-a", "--all", required=False, nargs='?', const=False, help="if will run all above")

args = vars(ap.parse_args())

if(args["all"] or args["augment"]):
    pass
if(args["all"] or args["preprocess"]):
    Preprocessor().run()
if(args["all"] or args["trainmodel"]):
    pass

