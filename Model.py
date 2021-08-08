from typing import Tuple
from matplotlib import pyplot
from numpy import load, zeros, ones
from numpy.random import randint
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam


def define_discriminator(image_shape:Tuple[int, int, int]) -> Model:
    initializer = RandomNormal(stddev=0.02)
    
    input_src_image = Input(shape=image_shape)
    input_target_image = Input(shape=image_shape)

    merged_images = Concatenate()([input_src_image, input_target_image])

    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(merged_images)
    d = LeakyReLU(alpha=0.2)(d)
	# C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=initializer)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=initializer)(d)

    patch_out = Activation('sigmoid')(d)
	# define model
    model = Model([input_src_image, input_target_image], patch_out)
	# compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])

    return model

def define_encoder_block(layer_in, n_filters:int, batchnorm:bool=True):
    initializer = RandomNormal(stddev=0.02)

    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(layer_in) 
    if batchnorm: g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    return g

def decoder_block(layer_in, skip_in, n_filters:int, dropout:bool=True):
    initializer = RandomNormal(stddev=0.02) 

    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout: g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)

    return g

def define_generator(image_shape:Tuple[int, int, int]=(256,256,3)) -> Model:
    initializer = RandomNormal(stddev=0.02)

    input_image = Input(shape=image_shape)

    # Encoder layers:
    e1 = define_encoder_block(input_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # Bottleneck of the U:
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(e7)
    b = Activation('relu')(b)
    # Decoder layers:
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(d7)
    
    output_image = Activation('tanh')(g)
    model = Model(input_image, output_image)

    return model

def define_gan(generator_model:Model, discriminator_model:Model, image_shape:Tuple[int, int, int]):
    for layer in discriminator_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    input_src = Input(shape=image_shape)
    # connect the source image to the generator input
    generator_output = generator_model(input_src)
    # connect the source input and generator output to the discriminator input
    discriminator_output = discriminator_model([input_src, generator_output])
    # src image as input, generated image and classification output
    model = Model(input_src, [generator_output, discriminator_output])

    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[1,100])
    
    return model


# load and prepare training images
def load_real_samples(filename:str):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5

	return [X1, X2]

# load and prepare training images
def load_real_samples(filename:str):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples:int, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(generator_model:Model, samples, patch_shape):
    # generate fake instance
    X = generator_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, generator_model:Model, dataset, n_samples:int=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(generator_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	generator_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


def train(discriminator_model:Model, generator_model:Model, gan_model:Model, dataset, n_epochs:int=100, n_batch:int=1) -> None:
	# determine the output square shape of the discriminator
	n_patch = discriminator_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(generator_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = discriminator_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = discriminator_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, generator_model, dataset)