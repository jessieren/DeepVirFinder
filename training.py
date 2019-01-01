#!/usr/bin/env python
# title             :training.py
# description       :Training deep learning for distinguishing viruses from hosts
# author            :Jie Ren renj@usc.edu
# date              :20180807
# version           :1.0
# usage             :python training.py -l 1000 -i ./train_example/tr/encode -j ./train_example/val/encode -o ./train_example/models -f 10 -n 1000 -d 1000 -e 50
# required packages :numpy, theano, keras, scikit-learn, Biopython
# conda create -n dvf python=3.6 numpy theano keras scikit-learn Biopython
#==============================================================================

#import multiprocessing
#os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True" 
#os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu0,floatX=float32" 
#os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
import numpy as np
import os
import sys
import random
import optparse
import keras
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import Average
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import h5py
import sklearn
from sklearn.metrics import roc_auc_score 

channel_num = 4

prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-l", "--len", action = "store", type = int, dest = "contigLength",
									help = "contig Length")
parser.add_option("-i", "--intr", action = "store", type = "string", dest = "inDirTr",
									default='./', help = "input directory for training data")
parser.add_option("-j", "--inval", action = "store", type = "string", dest = "inDirVal",
									default='./', help = "input directory for validation data")
                  # "/auto/cmb-12/fs3/renj/DL/data/year2014/encode"
parser.add_option("-o", "--out", action = "store", type = "string", dest = "outDir",
									default='./', help = "output directory")
parser.add_option("-f", "--fLen1", action = "store", type = int, dest = "filter_len1",
									help = "the length of filter")
parser.add_option("-n", "--fNum1", action = "store", type = int, dest = "nb_filter1",
									default=0, help = "number of filters in the convolutional layer")
parser.add_option("-d", "--dense", action = "store", type = int, dest = "nb_dense",
									default=0, help = "number of neurons in the dense layer")
parser.add_option("-e", "--epochs", action = "store", type = int, dest = "epochs",
									default=0, help = "number of epochs")

(options, args) = parser.parse_args()
if (options.contigLength is None or
		options.filter_len1 is None or
    options.nb_filter1 is None or options.nb_dense is None ) :
	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
	filelog.write(prog_base + ": ERROR: missing required command-line argument")
	parser.print_help()
	sys.exit(0)


contigLength = options.contigLength
filter_len1 = options.filter_len1
nb_filter1 = options.nb_filter1
nb_dense = options.nb_dense
inDirTr = options.inDirTr
inDirVal = options.inDirVal
outDir = options.outDir
if not os.path.exists(outDir):
    os.makedirs(outDir)
epochs = options.epochs


contigLengthk = contigLength/1000
if contigLengthk.is_integer() :
    contigLengthk = int(contigLengthk)

contigLengthk = str(contigLengthk)

teSampleRate = 0.1
valSampleRate = 0.1
rdseed = 0
random.seed(rdseed)

######## loading data for training, validation, testing ##########
print("...loading data...")

## phage RefSeq
print("...loading virus data...")
# training
filename_codetrfw = [ x for x in os.listdir(inDirTr) if 'codefw.npy' in x and 'virus' in x and contigLengthk + 'k' in x ][0]
print("data for training " + filename_codetrfw)
phageRef_codetrfw = np.load(os.path.join(inDirTr, filename_codetrfw))
phageRef_codetrbw = np.load(os.path.join(inDirTr, filename_codetrfw.replace('fw', 'bw')))
# validation
filename_codevalfw = [ x for x in os.listdir(inDirVal) if 'codefw.npy' in x and 'virus' in x and contigLengthk + 'k' in x ][0]
print("data for validation " + filename_codevalfw)
phageRef_codevalfw = np.load(os.path.join(inDirVal, filename_codevalfw))
phageRef_codevalbw = np.load(os.path.join(inDirVal, filename_codevalfw.replace('fw', 'bw')))


## host RefSeq
print("...loading host data...")
# training
filename_codetrfw = [ x for x in os.listdir(inDirTr) if 'codefw.npy' in x and 'host' in x and contigLengthk + 'k' in x ][0]
print("data for training " + filename_codetrfw)
hostRef_codetrfw = np.load(os.path.join(inDirTr, filename_codetrfw))
hostRef_codetrbw = np.load(os.path.join(inDirTr, filename_codetrfw.replace('fw', 'bw')))
# validation
filename_codevalfw = [ x for x in os.listdir(inDirVal) if 'codefw.npy' in x and 'host' in x and contigLengthk + 'k' in x ][0]
print("data for validation " + filename_codevalfw)
hostRef_codevalfw = np.load(os.path.join(inDirVal, filename_codevalfw))
hostRef_codevalbw = np.load(os.path.join(inDirVal, filename_codevalfw.replace('fw', 'bw')))


######## combine V and H, shuf training data ##########
print("...combining V and H...")
### training V+B
Y_tr = np.concatenate((np.repeat(0, hostRef_codetrfw.shape[0]), np.repeat(1, phageRef_codetrfw.shape[0])))
X_trfw = np.concatenate((hostRef_codetrfw, phageRef_codetrfw), axis=0)
del hostRef_codetrfw, phageRef_codetrfw
X_trbw = np.concatenate((hostRef_codetrbw, phageRef_codetrbw), axis=0)
del hostRef_codetrbw, phageRef_codetrbw
print("...shuffling training data...")
#size, seq_len, channel_num = X_tr.shape
index_trfw = list(range(0, X_trfw.shape[0]))
np.random.shuffle(index_trfw)
X_trfw_shuf = X_trfw[np.ix_(index_trfw, range(X_trfw.shape[1]), range(X_trfw.shape[2]))]
del X_trfw
X_trbw_shuf = X_trbw[np.ix_(index_trfw, range(X_trbw.shape[1]), range(X_trbw.shape[2]))]
del X_trbw
Y_tr_shuf = Y_tr[index_trfw]

### validation V+B
Y_val = np.concatenate((np.repeat(0, hostRef_codevalfw.shape[0]), np.repeat(1, phageRef_codevalfw.shape[0])))
X_valfw = np.concatenate((hostRef_codevalfw, phageRef_codevalfw), axis=0)
del hostRef_codevalfw, phageRef_codevalfw
X_valbw = np.concatenate((hostRef_codevalbw, phageRef_codevalbw), axis=0)
del hostRef_codevalbw, phageRef_codevalbw


######### training model #############
# parameters
POOL_FACTOR = 1
dropout_cnn = 0.1
dropout_pool = 0.1
dropout_dense = 0.1
learningrate = 0.001
batch_size=int(X_trfw_shuf.shape[0]/(1000*1000/contigLength)) ## smaller batch size can reduce memory
pool_len1 = int((contigLength-filter_len1+1)/POOL_FACTOR)

modPattern = 'model_siamese_varlen_'+contigLengthk+'k_fl'+str(filter_len1)+'_fn'+str(nb_filter1)+'_dn'+str(nb_dense)
#modName = os.path.join( outDir, modPattern +'_ep{epoch:02d}_acc{acc:.2f}'+'.h5')
modName = os.path.join( outDir, modPattern + '.h5')
checkpointer = keras.callbacks.ModelCheckpoint(filepath=modName, verbose=1,save_best_only=True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1)

##### build model #####

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output


print("...building model...")
## if model exists
if os.path.isfile(modName):
  model = load_model(modName)
  print("...model exists...")
else :
  ## siamese
  forward_input = Input(shape=(None, channel_num))
  reverse_input = Input(shape=(None, channel_num))
  hidden_layers = [
      Conv1D(filters = nb_filter1, kernel_size = filter_len1, activation='relu'),
      GlobalMaxPooling1D(),
      # https://github.com/fchollet/keras/issues/1920
      # https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
      Dropout(dropout_pool),
      Dense(nb_dense, activation='relu'),
      Dropout(dropout_dense),
      Dense(1, activation='sigmoid')
  ]
  forward_output = get_output(forward_input, hidden_layers)     
  reverse_output = get_output(reverse_input, hidden_layers)
  output = Average()([forward_output, reverse_output])
  model = Model(inputs=[forward_input, reverse_input], outputs=output)
  model.compile(Adam(lr=learningrate), 'binary_crossentropy', metrics=['accuracy'])

print("...fitting model...")
print(contigLengthk+'k_fl'+str(filter_len1)+'_fn'+str(nb_filter1)+'_dn'+str(nb_dense)+'_ep'+str(epochs))
model.fit(x = [X_trfw_shuf, X_trbw_shuf], y = Y_tr_shuf, \
            batch_size=batch_size, epochs=epochs, verbose=2, \
            validation_data=([X_valfw, X_valbw], Y_val), \
            callbacks=[checkpointer, earlystopper])

            
            
## Final evaluation AUC ###

## train data
type = 'tr'
X_fw = X_trfw_shuf
X_bw = X_trbw_shuf
Y = Y_tr_shuf
print("...predicting "+type+"...\n")
Y_pred = model.predict([X_fw, X_bw], batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y, Y_pred)
print('auc_'+type+'='+str(auc)+'\n')
#np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_pred.txt'), np.transpose(Y_pred))
#np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_true.txt'), np.transpose(Y))

del Y, X_fw, X_bw


# val data
type = 'val'
X_fw = X_valfw
X_bw = X_valbw
Y = Y_val
print("...predicting "+type+"...\n")
Y_pred = model.predict([X_fw, X_bw], batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y, Y_pred)
print('auc_'+type+'='+str(auc)+'\n')
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_pred.txt'), np.transpose(Y_pred))
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_true.txt'), np.transpose(Y))

del Y, X_fw, X_bw



