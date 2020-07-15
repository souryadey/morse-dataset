# morse-dataset
#### Generate synthetic datasets for Morse code symbol classification for machine learning algotihms such as artificial neural networks.
#### Compute the inherent difficulty of the classification problem on these datasets using different metrics.

<br>This **research paper** has more details. Please consider citing it if you use or benefit from this work:<br>
S. Dey, K. M. Chugg and P. A. Beerel, “Morse Code Datasets for Machine Learning,” in _9th International Conference on Computing, Communication and Networking Technologies (ICCCNT)_, pp. 1-7, Jul 2018. (Won Best Paper Award)<br>
Available on [IEEE](https://ieeexplore.ieee.org/document/8494011) and [arXiv](https://arxiv.org/abs/1807.04239) (copyright owned by IEEE).

<br>For a short description of dataset generation, see Sourya Dey's [blog post](https://cobaltfolly.wordpress.com/2017/10/15/morse-code-dataset-for-artificial-neural-networks)<br>
For a guide to Morse code symbols, see [morse_tree](./morse_tree.png). Go left for dots and right for dashes.

This family of datasets now has its own [IEEEDataPort page](https://ieee-dataport.org/open-access/morse-code-symbol-classification) and [competition](http://ieee-dataport.org/competitions/morse-code-symbol-classification) (open till October 14, 2020!)

<br>**Requirements**: Python 3, numpy, scipy

<br>Description of [generate_morse_dataset.py](./generate_morse_dataset.py):

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
- baseline.npz : Uses default parameters
- difficult.npz : Uses noisesd=4, leadingsp_rand=1, minlendash=3

Use `load_data` to extract the data and labels into training, validation and test:
```
xtrain, ytrain, xval, yval, xtest, ytest = load_data(filename = './baseline.npz')
```

Run [dataset_metrics](./dataset_metrics.py) to test dataset difficulty, for example:
```
L, U, D, T = dataset_metrics('./baseline.npz')
```

