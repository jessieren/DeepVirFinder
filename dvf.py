#!/usr/bin/env python
# title             :dvf.py
# description       :Identifying viral sequences from metagenomic data by deep learning
# author            :Jie Ren renj@usc.edu
# date              :20180807
# version           :1.0
# usage             :python dvf.py -i <path_to_input_fasta> -o <path_to_output_directory>
# required packages :numpy, theano, keras
# conda create -n dvf python=3.6 numpy theano keras scikit-learn Biopython
#==============================================================================


#### Step 0: pass arguments into the program ####
import os, sys, optparse, warnings
import gzip
import bz2
from Bio.Seq import Seq

import h5py
import concurrent.futures

import pandas as pd
import numpy as np

import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

os.environ['KERAS_BACKEND'] = 'theano'

#### Step 0: import keras libraries ####
import keras
from keras.models import load_model
#sys.setrecursionlimit(10000000)
#os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True"
#os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu0,floatX=float32"
#os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

prog_base = os.path.split(sys.argv[0])[1]
parser = optparse.OptionParser()

parser.add_option(
    "-i", "--in", action = "store", type = "string", dest = "input_fa",
    help = "input fasta file, support gzip and bzip2 format")

parser.add_option(
    "-m", "--mod", action = "store", type = "string", dest = "modDir",
	default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    help = "model directory (default ./models)")

parser.add_option(
    "-o", "--out", action = "store", type = "string", dest = "output_dir",
	default='./', help = "output directory")

parser.add_option(
    "-l", "--len", action = "store", type = "int", dest = "cutoff_len",
	default=1, help = "predict only for sequence >= L bp (default 1)")

parser.add_option(
    "-c", "--core", action = "store", type = "int", dest = "core_num",
	default=1, help = "number of parallel cores (default 1)")

(options, args) = parser.parse_args()
if (options.input_fa is None) :
	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
	parser.print_help()
	sys.exit(0)

input_fa = options.input_fa
if options.output_dir != './' :
    output_dir = options.output_dir
else :
    output_dir = os.path.dirname(os.path.abspath(input_fa))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cutoff_len = options.cutoff_len
core_num = options.core_num


#### Step 0: function for encoding sequences into matrices of size 4 by n ####
def encodeSeq(seq) :
    seq_code = list()
    for pos in range(len(seq)) :
        letter = seq[pos]
        if letter in ['A', 'a'] :
            code = [1,0,0,0]
        elif letter in ['C', 'c'] :
            code = [0,1,0,0]
        elif letter in ['G', 'g'] :
            code = [0,0,1,0]
        elif letter in ['T', 't'] :
            code = [0,0,0,1]
        else :
            code = [1/4, 1/4, 1/4, 1/4]
        seq_code.append(code)
    return seq_code

#### Step 0: function for predicting viral score using the trained model ####
def pred(code_tuple):
    head = code_tuple[0]
    codefw = code_tuple[1]
    codebw = code_tuple[2]

    #print('predicting ' + head)
    seqL = len(codefw)

    if seqL < 300 :
        model = modDict['0.15']
        null = nullDict['0.15']
    elif seqL < 500 and seqL >= 300 :
        model = modDict['0.3']
        null = nullDict['0.3']
    elif seqL < 1000 and seqL >= 500 :
        model = modDict['0.5']
        null = nullDict['0.5']
    else :
        model = modDict['1']
        null = nullDict['1']

    score = model.predict([np.array([codefw]), np.array([codebw])], batch_size=1)
    pvalue = sum([x>score for x in null])/len(null)

    #print('predicting ' + head + ' done')
    return [head, seqL, score, pvalue]


#### Step 1: load model ####
print("1. Loading Models.")
#modDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
modDir = options.modDir
print("   model directory {}".format(modDir))

modDict = {}
nullDict = {}

warnings.filterwarnings('ignore', 'Error in loading the saved optimizer ')

for contigLengthk in ['0.15', '0.3', '0.5', '1'] :
    modPattern = 'model_siamese_varlen_'+contigLengthk+'k'
    modName = [ x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5") ][0]

    from pprint import pprint
    pprint(modName)

    #model_1000 = load_model(os.path.join(modDir, modName))
    modDict[contigLengthk] = load_model(os.path.join(modDir, modName))
    Y_pred_file = [ x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x ][0]
    with open(os.path.join(modDir, Y_pred_file)) as f:
        tmp = [line.split() for line in f][0]
        Y_pred = [float(x) for x in tmp ]
    Y_true_file = [ x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x ][0]
    with open(os.path.join(modDir, Y_true_file)) as f:
        tmp = [ line.split()[0] for line in f]
        Y_true = [ float(x) for x in tmp ]
    nullDict[contigLengthk] =  Y_pred[:Y_true.index(1)]


#### Step2 : encode sequences in input fasta, and predict scores ####

# clean the output file
print("2. Encoding and Predicting Sequences.")
code_list = []
dvf_res_list = []
head = ''
lineNum = 0
seq = ''
flag = 0
total_len = 0
total_seq_num = 0

if input_fa.endswith(".gz"):
    faLines = gzip.open(input_fa, 'rt')
elif input_fa.endswith(".bz2"):
    faLines = bz2.open(input_fa, 'rt')
else:
    faLines = open(input_fa, 'r')

for line in faLines :
    #print(line)
    lineNum += 1

    if flag == 0 and line[0] == '>' :
        print("   processing line " + str(lineNum))
        head = line.strip()[1:]
        continue
    elif line[0] != '>' :
        seq = seq + line.strip()
        flag += 1
    elif flag > 0 and line[0] == '>' :
        countN = seq.count("N")
        if (countN / len(seq)) <= 0.3 and (len(seq) >= cutoff_len) and (len(seq) < 2100000):
            #print(head)
            #print("encode seq")
            codefw = encodeSeq(seq)

            #print("reverse complement seq")
            seqR = Seq(seq).reverse_complement()

            #print("encode rc seqR")
            codebw = encodeSeq(seqR)

            code_list.append((head, codefw, codebw))
            total_len += len(seq)
            total_seq_num += 1

            if len(code_list) % 100 == 0 :
            #if total_len > 100000:
            #if len(code_list) > 0:
                print("   total len: " + str(total_len))
                print("   processing seqs num " + str(total_seq_num))
                print("   processing line " + str(lineNum))

                print("multiprocessing ")
                with concurrent.futures.ProcessPoolExecutor(max_workers=core_num) as executor:
                    for res in executor.map(pred, code_list):
                        dvf_res_list.append(res)
                print("multiprocessing done\n")

                code_list = []
                total_len = 0
        else :
            if countN / len(seq) > 0.3 :
                print("   {} has >30% Ns, skipping it".format(head))
            # else :
            #    print("   {} < {}bp, skipping it".format(head, cutoff_len))

        flag = 0
        seq = ''
        head = line.strip()[1:]

if flag > 0 :
    countN = seq.count("N")
    if countN/len(seq) <= 0.3 and len(seq) >= cutoff_len :
        codefw = encodeSeq(seq)
        seqR = Seq(seq).reverse_complement()
        codebw = encodeSeq(seqR)
        code_list.append((head, codefw, codebw))
        total_len += len(seq)
        total_seq_num += 1

    print("   total_len: " + str(total_len))
    print("   processing seqs num " + str(total_seq_num))
    print("   processing line " + str(lineNum))

    with concurrent.futures.ProcessPoolExecutor(max_workers=core_num) as executor:
        for res in executor.map(pred, code_list):
            dvf_res_list.append(res)
    print("multiprocessing done")

faLines.close()

outfile = os.path.join(
    output_dir,
    os.path.basename(input_fa) + '_gt' + str(cutoff_len) + 'bp_dvfpred.txt')

pd.DataFrame(dvf_res_list, columns=["name", "len", "score", "pvalue"])\
.to_csv(outfile, sep="\t", index=False)


print("3. Done. Thank you for using DeepVirFinder.")
print("   output in {}".format(outfile))
