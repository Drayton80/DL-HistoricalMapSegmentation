# example of pix2pix gan for satellite to map image-to-image translation
from typing import Tuple
from matplotlib import pyplot
from numpy import load, zeros, ones, ndarray
from numpy.random import randint
from numpy.lib.npyio import NpzFile
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# define the discriminator model
def define_discriminator(image_shape:Tuple[int, int, int]) -> Model:
    kernel_initializer = RandomNormal(stddev=0.02)
    input_source_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    
    # concatenate images channel-wise
    merged = Concatenate()([input_source_image, in_target_image])
   
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=kernel_initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=kernel_initializer)(d)
    patch_out = Activation('sigmoid')(d)

    # define model
    model = Model([input_source_image, in_target_image], patch_out)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
    
    return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3)) -> Model:
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape) -> Model:
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data:NpzFile = load(filename)
	# unpack arrays, X1 is the original maps and X2 is the road lines:
	X1:ndarray = data['arr_0']
	X2:ndarray = data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	source_dataset:ndarray = dataset[0]
	target_dataset:ndarray = dataset[1]
	# choose random instances
	random_indexes:list[int, int, int] = randint(0, source_dataset.shape[0], n_samples)
	# retrieve selected images
	source_samples:ndarray = source_dataset[random_indexes]
	target_samples:ndarray = target_dataset[random_indexes]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [source_samples, target_samples], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def summarize_file_name(step: int, epoch: int):
	trained_id:str = 'step' + str(step) + '_epoch' + str(epoch)
	trained_dir:str = 'trained models/'

	return trained_dir + trained_id

# generate samples and save as a plot and save the model
def save_trained_preview(epoch, step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# Saving the plot preview in to file:
	pyplot.savefig(summarize_file_name(step, epoch) + '_preview.png')
	pyplot.close()
	print('> Saved preview: ' + summarize_file_name(step, epoch))

def save_trained_model(epoch:int, step:int, g_model:Model):
	# Saving the model in a HDF5 file:
	g_model.save(summarize_file_name(step, epoch) + '_model.h5')
	print('> Saved model: ' + summarize_file_name(step, epoch))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=300, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations necessary to complete all the epochs
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	epoch:int = 0
	
	Path('trained models/').mkdir(parents=True, exist_ok=True)

	for step in range(n_steps):
		# select a batch of real samples
		[real_map, real_mask], real_y = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		fake_mask, fake_y = generate_fake_samples(g_model, real_map, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([real_map, real_mask], real_y)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([real_map, fake_mask], fake_y)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(real_map, [real_y, real_mask])
		# summarize performance
		print('> step[%d] epoch[%d] - losses: dis_real[%.3f] dis_fake[%.3f] generator[%.3f]' % (step+1, epoch, d_loss1, d_loss2, g_loss))
		# checks if the epoch had finish
		if (step+1) % bat_per_epo == 0:
			epoch = epoch + 1
		# save trained preview after 10 epochs
		if (step+1) % (bat_per_epo * 10) == 0:
			save_trained_preview(epoch, step, g_model, dataset)
		# save trained model after 100 epochs
		if (step+1) % (bat_per_epo * 50) == 0:
			save_trained_model(epoch, step, g_model)

def setup_and_train(image_file:str) -> None:
	# load image data
	dataset = load_real_samples(image_file)
	print('Loaded', dataset[0].shape, dataset[1].shape)
	# define input shape based on the loaded dataset
	image_shape:Tuple[int, int, int] = dataset[0].shape[1:]
	# define the models
	discriminator_model:Model = define_discriminator(image_shape)
	generator_model:Model = define_generator(image_shape)
	# define the composite model
	gan_model:Model = define_gan(generator_model, discriminator_model, image_shape)
	# train model
	train(discriminator_model, generator_model, gan_model, dataset)