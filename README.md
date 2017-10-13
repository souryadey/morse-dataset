# morse-dataset
Generate Morse code datasets for training artificial neural networks

For a complete description and motivation, refer to my blog: 

Here is a more succinct description of the Python script:


Style='BW': Black and white having 0s for spaces and 1s for dot or dash. Noise means bit flips
Style='GRAY': Gaussian grayscale levels. Noise is additive Gaussian

Framelen: Total length of frame for 1 character, i.e. 1 input
Classes: How many characters, i.e. outputs
TReach,VAeach,TEeach: No. of training, validation and test cases FOR EACH class
minlendot,maxlendot,... : min and max length of dots, dashes and intermediate spaces
leadingsp_rand: Set to 0 to have no leading spaces, otherwise 1 to have random number of leading spaces

Black and white only:
    maxflip: Noise measure. Max how many bits to flip
    SET maxfilp=0 for NO NOISE

Grayscale only:
    levels: How many levels for dots and dashes. Will be normalized at end
    symbmean: Mean level for any symbol (dot or dash)
    symbsd: Standard deviation for symbol levels
    noisemean: Mean level for noise
    noisesd: Standard deviation for noise
    SET noisesd=0 for NO NOISE
	

An already generated dataset using the default grayscale parameters is included as morse_dataset.npz. Use these commands to extract the data and labels into training, validation and test:

	data = np.load('./morse_dataset.npz')
	xtrain = data['xtr']
	ytrain = data['ytr']
	xval = data['xva']
	yval = data['yva']
	xtest = data['xte']
	ytest = data['yte']