#==============================================================================
# Created by Sourya Dey, USC
#==============================================================================

import numpy as np


def store_morse_data(Style='GRAY', save_filename = '',
                     Framelen = 64, dilation=1, Classes = 64,
                     TReach = 5000, VAeach = 1000, TEeach = 1000,
                     minlendot = 1, maxlendot = 3, minlendash = 4, maxlendash = 9, minlensp = 1, maxlensp = 3, leadingsp_rand=0,
                     maxflip = 0,
                     levels = 16, symbmean = 12, symbsd = 1.34, noisemean = 0, noisesd = 0):
    '''
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
    '''
    Codebook = np.load('./Codebook.npy').item()
    assert len(Codebook)==Classes, 'Number of classes inconsistent'

    TOTeach = TReach+VAeach+TEeach
    TRcases = TReach*Classes
    VAcases = VAeach*Classes
    TEcases = TEeach*Classes
    TOTcases = TOTeach*Classes

    Framelen *= dilation
    minlendot *= dilation
    maxlendot *= dilation
    minlendash *= dilation
    maxlendash *= dilation
    minlensp *= dilation
    maxlensp *= dilation

    xdata = np.zeros((TOTcases,Framelen),dtype='float')
    ydata = np.zeros((TOTcases,Classes),dtype='float')
    index = 0

    for n in range(len(Codebook)):
        code = Codebook.values()[n]

        for repeat in range(TOTeach):
            ydata[index][n] = 1

            if Style=='BW':
                bitcode = []
                for c in range(len(code)): #parse through dots and dashes
                    if code[c]=='.':
                        bitcode.extend([1 for i in range(np.random.randint(minlendot,maxlendot+1))])
                    else:
                        bitcode.extend([1 for i in range(np.random.randint(minlendash,maxlendash+1))])
                    if c != len(code)-1: #don't add space after last dot or dash
                        bitcode.extend([0 for i in range(np.random.randint(minlensp,maxlensp+1))])

            elif Style=='GRAY':
                bitcode = np.array(())
                for c in range(len(code)): #parse through dots and dashes
                    if code[c]=='.':
                        bitcode = np.append(bitcode, symbsd*np.random.randn(np.random.randint(minlendot,maxlendot+1))+symbmean)
                    else:
                        bitcode = np.append(bitcode, symbsd*np.random.randn(np.random.randint(minlendash,maxlendash+1))+symbmean)
                    if c != len(code)-1: #don't add space after last dot or dash
                        bitcode = np.append(bitcode, np.zeros(np.random.randint(minlensp,maxlensp+1)))

            ## Leading and trailing spaces
            ltsp = Framelen - len(bitcode) #total number of spaces required
            if leadingsp_rand==1:
                lsp = [0 for i in range(np.random.randint(ltsp+1))] #leading spaces
                tsp = [0 for i in range(ltsp - len(lsp))] #trailing spaces
            else:
                lsp = []
                tsp = [0 for i in range(ltsp)]
            xdata[index] = np.concatenate((np.asarray(lsp),np.asarray(bitcode),np.asarray(tsp)))

            ## Noise
            if Style=='BW':
                numflip = np.random.randint(maxflip+1) #how many bits to flip
                bitstoflip = np.random.choice(Framelen,size=numflip,replace=False)
                xdata[index][bitstoflip] = 1-xdata[index][bitstoflip]

            elif Style=='GRAY':
                xdata[index] += noisesd*np.random.randn(Framelen)+noisemean

            index += 1

        print('{0}/{1}'.format(n+1,Classes)) #progress...

    ### Post-processing
    if Style=='GRAY':
        levels = float(levels)
        #xdata = np.around(xdata) #Round before so that final xdata has values = n/16, n in [0,16]
        xdata[xdata>levels] = levels
        xdata[xdata<0] = 0.
        xdata /= levels #set range to 0-1
        xdata = np.around(xdata, 3) #Round after

    ### Split into training, validation, test
    xtr = np.zeros((TRcases,Framelen))
    ytr = np.zeros((TRcases,Classes))
    xva = np.zeros((VAcases,Framelen))
    yva = np.zeros((VAcases,Classes))
    xte = np.zeros((TEcases,Framelen))
    yte = np.zeros((TEcases,Classes))
    for i in range(Classes):
        xtr[i*TReach:(i+1)*TReach,:] = xdata[i*TOTeach:i*TOTeach+TReach,:]
        ytr[i*TReach:(i+1)*TReach] = ydata[i*TOTeach:i*TOTeach+TReach]
        xva[i*VAeach:(i+1)*VAeach] = xdata[i*TOTeach+TReach:i*TOTeach+TReach+VAeach]
        yva[i*VAeach:(i+1)*VAeach] = ydata[i*TOTeach+TReach:i*TOTeach+TReach+VAeach]
        xte[i*TEeach:(i+1)*TEeach] = xdata[i*TOTeach+TReach+VAeach:(i+1)*TOTeach]
        yte[i*TEeach:(i+1)*TEeach] = ydata[i*TOTeach+TReach+VAeach:(i+1)*TOTeach]

    ### Shuffle
    n = np.random.permutation(TRcases)
    xtr = xtr[n[:]]
    ytr = ytr[n[:]]
    n = np.random.permutation(VAcases)
    xva = xva[n[:]]
    yva = yva[n[:]]
    n = np.random.permutation(TEcases)
    xte = xte[n[:]]
    yte = yte[n[:]]

    ### Save
    if save_filename!='':
        np.savez_compressed(save_filename, xtr=xtr,ytr=ytr,xva=xva,yva=yva,xte=xte,yte=yte)
    else:
        if Style=='BW':
            np.savez_compressed('./morseBW{0}_{1}i{2}o_noise{3}_lsp{4}.npz'.format(TOTcases,Framelen,Classes,maxflip,leadingsp_rand), xtr=xtr,ytr=ytr,xva=xva,yva=yva,xte=xte,yte=yte)
        elif Style=='GRAY':
            np.savez_compressed('./morseGRAY{0}_{1}i{2}o_noisesd{3}_lsp{4}_dash{5}{6}.npz'.format(TOTcases,Framelen,Classes,noisesd,leadingsp_rand,minlendash,maxlendash), xtr=xtr,ytr=ytr,xva=xva,yva=yva,xte=xte,yte=yte)


#==============================================================================
# Get baseline dataset
#==============================================================================
store_morse_data(save_filename='./baseline.npz')

#==============================================================================
# Get a really difficult dataset with
# noise, leading spaces, and dashes confused as dots and spaces
#==============================================================================
store_morse_data(save_filename='./difficult.npz', minlendash=3, leadingsp_rand=1, noisesd=4)


