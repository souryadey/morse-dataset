# morse-dataset
Generate Morse code datasets for machine learning algotihms such as artificial neural networks

Complete description and motivation soon to follow after I finish writing a paper on this. My blog has a fairly nice description : 

Here is a more succinct description of the Python script:

	Create morse code datasets:
	        Style='BW': Black and white having 0s for spaces and 1s for dot or dash. Noise means bit flips
	        Style='GRAY': Gaussian grayscale levels. Noise is additive Gaussian
	        save_filename: Leave as '' to save in preset path inside

	    Framelen: Total length of frame for 1 character
	    Classes: How many characters
	    TReach,VAeach,TEeach: No. of training, validation and test cases FOR EACH class
	    minlendot,maxlendot,... : min and max length of dots, dashes and intermediate spaces
	    leadingsp_rand: Set to 0 to have no leading spaces, otherwise 1 to have random number of leading spaces
	    dilation: If >1, all lengths are increased by this factor

	    Black and white only:
	        maxflip: Noise measure. Max how many bits to flip
	        SET maxflip=0 for NO NOISE
	    Grayscale only:
	        levels: How many levels for dots and dashes. Will be normalized at end
	        symbmean: Mean level for any symbol (dot or dash)
	        symbsd: Standard deviation for symbol levels
	        noisemean: Mean level for noise
	        noisesd: Standard deviation for noise
	        SET noisesd=0 for NO NOISE
	

2 already generated datasets are included:
	baseline.npz : Uses default parameters
	difficult.npz : Uses noisesd=4, leadingsp_rand=1, minlendash=3
Use these commands to extract the data and labels into training, validation and test:

	data = np.load('./baseline.npz')
	xtrain = data['xtr']
	ytrain = data['ytr']
	xval = data['xva']
	yval = data['yva']
	xtest = data['xte']
	ytest = data['yte']
	

For a guide to Morse code symbols, see morse_tree.png
Go left for dots and right for dashes