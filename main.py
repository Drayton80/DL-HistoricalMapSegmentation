import argparse
import preprocessor
import oldModel as modeler
import augmentor
import validator

ap = argparse.ArgumentParser()
ap.add_argument("-ag", "--augment", required=False, nargs='?', const=False, help="if the data will be augmented or not")
ap.add_argument("-p", "--preprocess", required=False, nargs='?', const=False, help="if the data will be preprocessed or not")
ap.add_argument("-tm", "--trainmodel", required=False, nargs='?', const=False, help="if the model will train or not")
ap.add_argument("-rm", "--runmodel", required=False, nargs='?', const=False, help="if the model will run or not")
ap.add_argument("-a", "--all", required=False, nargs='?', const=False, help="if will run all above")

args = vars(ap.parse_args())

if(args["all"] or args["augment"]):
    augmentor.run()
if(args["all"] or args["preprocess"]):
    preprocessor.run(compress_in='dataset_train.npz')
if(args["all"] or args["trainmodel"]):
    modeler.setup_and_train('dataset_train.npz')

validator.predict_test_images('trained models/previews 2/step98699_epoch300_model.h5')

#modeler.generate('maps/preprocessed/0/original/17.png', '001620')
#modeler.generate('maps/original/maps-of-medieval-cities-bologna.jpg', '001620')

