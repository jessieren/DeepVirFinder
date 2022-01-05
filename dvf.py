#!/usr/bin/env python
# title             :dvf.py
# description       :Identifying viral sequences from metagenomic data by deep learning
# author            :Jie Ren renj@usc.edu
# corrections       :Jean-Sebastien Gounot jsgounot@gmail.com
# date              :20210105
# version           :2.0
# usage             :python dvf.py -i <path_to_input_fasta> -o <path_to_output_directory>
# required packages :numpy, theano, keras 
# conda create -n dvf python=3.6 numpy theano keras scikit-learn Biopython
#==============================================================================


#### Step 0: pass arguments into the program ####
import os, sys, time
import optparse, warnings
import random

import h5py, multiprocessing
import numpy as np

os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import load_model

class FRecord():

    COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}

    ENCODING = {
        'A': [1,0,0,0],
        'C': [0,1,0,0],
        'G': [0,0,1,0],
        'T': [0,0,0,1],
    }

    MISSING = [1/4, 1/4, 1/4, 1/4]

    def __init__(self, name, record):
        self.name = name
        self.record = record

    def seqlen(self):
        return len(self.record)

    def ncount(self):
        return self.record.count("N")

    def complement(self):
        nrecord = ''.join(FRecord.COMPLEMENT.get(base, base) for base in reversed(self.record))
        return FRecord(self.name, nrecord)

    def encode(self):
        record = self.record.upper()
        return np.array([[FRecord.ENCODING.get(base, FRecord.MISSING)
            for base in record]])

    @staticmethod
    def from_fasta(fname):
        name, record = None, None
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if name and record:
                        yield FRecord(name, record)

                    name = line[1:]
                    record = ''

                else:
                    record += line

        yield FRecord(name, record)

def load_models(modDir):
    #### Step 1: load model ####
    print("1. Loading Models. Model directory {}".format(modDir))

    modDict = {}
    nullDict = {}

    warnings.filterwarnings('ignore', 'Error in loading the saved optimizer ')

    for contigLengthk in ['0.15', '0.3', '0.5', '1'] :
        modPattern = 'model_siamese_varlen_' + contigLengthk + 'k'
        modName = [x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5")][0]
        modDict[contigLengthk] = load_model(os.path.join(modDir, modName))
  
        ypred_file = [x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x]
        assert len(ypred_file) == 1
        ypred_file = os.path.join(modDir, ypred_file[0])
        ypred = np.loadtxt(ypred_file, delimiter=' ')

        ytrue_file = [x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x]
        assert len(ytrue_file) == 1
        ytrue_file = os.path.join(modDir, ytrue_file[0])
        ytrue = np.loadtxt(ytrue_file, delimiter=' ')

        index_one = np.where(ytrue == 1)[0][0]
        nullDict[contigLengthk] = ypred[:index_one]        

    return modDict, nullDict

def predict(frecord, modDict, nullDict):
    seqL = frecord.seqlen()

    if seqL < 300:
        key = '0.15'
    elif seqL < 500 and seqL >= 300 :
        key = '0.3'
    elif seqL < 1000 and seqL >= 500 :
        key = '0.5'
    else :
        key = '1'
    
    model = modDict[key]
    null = nullDict[key]

    codefw = frecord.encode()
    codebw = frecord.complement().encode()
    score = model.predict([codefw, codebw], batch_size=1)

    if score.shape != (1, 1):
        raise Exception('Uncorrect score shape', score.shape)
    
    score = float(score)
    pvalue = len(null[null > score]) / len(null)

    if random.randint(0, 1000) == 1:
        print ('checking with old pvalue ... ')
        oldpvalue = sum([x>score for x in null]) / len(null)
        assert oldpvalue == pvalue
    
    return (frecord.name, seqL, score, float(pvalue))

def process(fdata, modDict, nullDict, cutoff_len):
    print("2. Encoding and Predicting Sequences.")

    for frecord in fdata:

        if frecord.seqlen() < cutoff_len:
            print ('{} does not pass cutoff len ({} - {}), pass ...'.format(frecord.name, frecord.seqlen(), cutoff_len))
            continue

        prcN = frecord.ncount() / frecord.seqlen()
        if prcN > 0.3:
            print ('{} N percentage to high ({} - 0.3), pass ...'.format(frecord.name, prcN))
            continue

        print ('Predict for {}'.format(frecord.name))
        yield predict(frecord, modDict, nullDict)

def write_output(outfile, results):
    with open(outfile, 'w') as f:
        header = '\t'.join(('name', 'len', 'score', 'pvalue')) + '\n'
        f.write(header)
        
        for result in results:
            result = [str(value) for value in result]
            f.write('\t'.join(result) + '\n')


if __name__ == '__main__':

    start_time = time.time()

    prog_base = os.path.split(sys.argv[0])[1]
    parser = optparse.OptionParser()
    
    parser.add_option("-i", "--in", action = "store", type = "string", dest = "input_fa",
        help = "input fasta file")
    
    parser.add_option("-m", "--mod", action = "store", type = "string", dest = "modDir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), 
        help = "model directory (default ./models)")
    
    parser.add_option("-o", "--out", action = "store", type = "string", dest = "output_dir",
        default='./', help = "output directory")
    
    parser.add_option("-l", "--len", action = "store", type = "int", dest = "cutoff_len",
        default=1, help = "predict only for sequence >= L bp (default 1)")

    parser.add_option("-f", "--flags", action = "store", type = "string", dest = "tflags",
        default='', help = "theanos flags")

    (options, args) = parser.parse_args()
    if (options.input_fa is None) :
        sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
        filelog.write(prog_base + ": ERROR: missing required command-line argument")
        parser.print_help()
        sys.exit(1)

    input_fa = options.input_fa

    if options.output_dir != './' :
      output_dir = options.output_dir
    else :
      output_dir = os.path.dirname(os.path.abspath(input_fa))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cutoff_len = options.cutoff_len

    if options.tflags:
        #os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True" 
        #os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu0,floatX=float32" 
        os.environ['THEANO_FLAGS'] = options.tflags

    modDict, nullDict = load_models(options.modDir)
    fdata = FRecord.from_fasta(input_fa)
    results = process(fdata, modDict, nullDict, cutoff_len)

    outfile = os.path.join(output_dir, os.path.basename(input_fa) + '_gt' + str(cutoff_len) + 'bp_dvfpred.txt')
    write_output(outfile, results)

    print("3. Done. Thank you for using DeepVirFinder. Output in {}".format(outfile))

    end_time = time.time()
    print(end_time - start_time, "seconds")