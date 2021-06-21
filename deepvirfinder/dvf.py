#!/usr/bin/env python
# title             :dvf.py
# description       :Identifying viral sequences from metagenomic data by deep learning
# author            :Jie Ren renj@usc.edu, @papanikos
# date              :20202308
# version           :1.0
# usage             :dvf.py -i <path_to_input_fasta> -o <path_to_output_directory>
# required packages :numpy, theano, keras 
# conda create -n dvf python=3.6 numpy theano keras scikit-learn Biopython
#==============================================================================
import os
import warnings
import pkg_resources
from pathlib import Path

import Bio.SeqIO as SeqIO
import argparse
import tqdm
import h5py
import multiprocessing as mp
import numpy as np

# Suppress warnings and output when importing libraries
warnings.filterwarnings('ignore', 'Error in loading the saved optimizer ')
warnings.filterwarnings('ignore', category=FutureWarning)

from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, 'w')):
    os.environ["KERAS_BACKEND"] = "theano"
    import keras
    from keras.models import load_model


def set_models_dir():
    """
    Utility function to set the models directory
    """
    install_location = Path(pkg_resources.get_distribution('deepvirfinder').location)
    models_dir = install_location.joinpath('deepvirfinder/models')
    if models_dir.exists():
        return models_dir
    else:
        return './models'


def parse_args():
    """
    Parse options, set defaults
    """
    parser = argparse.ArgumentParser(description="Identifying viral sequences from "
                                     "metagenomic data by deep learning")

    # Re-organize argument groups
    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group("Required arguments")

    requiredArgs.add_argument(
        "-i", "--in",
        required=True,
        type = str,
        dest = "input_fa",
        help = "input fasta file"
        )

    optionalArgs.add_argument(
        "-m", "--mod",
        type = str,
        dest = "modDir",
        default=set_models_dir(),
        help = "models directory (default : %(default)s)"
        )

    optionalArgs.add_argument(
        "-o", "--out",
        type = str,
        dest = "output_dir",
        default='./',
        help = "output directory (default : %(default)s)"
        )

    optionalArgs.add_argument(
        "-l", "--min-len",
        type = int,
        default = 0,
        help = "Minimum length of sequence to analyze (default : %(default)s)"
        )

    optionalArgs.add_argument(
        "-L", "--max-len",
        type = int,
        default = 1000000,
        help = "Maximum length of sequence to analyze (default : %(default)s)"
        )

    optionalArgs.add_argument(
        "-b", "--batch-size",
        type = int,
        default = 10,
        help = "Processing batch size (default : %(default)s)"
    )

    optionalArgs.add_argument(
        "-c", "--cores",
        type = int,
        dest = "core_num",
        default= 1,
        help = "number of parallel cores (default: %(default)s)"
        )

    # Put the optionalArgs back
    parser._action_groups.append(optionalArgs)

    return parser.parse_args()


def encodeSeq(seq):
    """
    Return a list of length == len(seq) where
    all bases of input sequence of are one hot encoded.

    ATCG -->  [[1,0,0,0],[0,0,0,1],[0,1,0,0], [0,0,1,0]]
    """
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


if __name__ == '__main__':
    args = parse_args()

    input_fa = args.input_fa
    if args.output_dir != './':
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.abspath(input_fa))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    min_len = args.min_len
    max_len = args.max_len
    core_num = args.core_num

    # set-up string tanslation for complementing DNA
    complement_table = str.maketrans('aAcCgGtTnN',
                                     'tTgGcCaAnN')

    print("1. Loading Models.")
    modDir = args.modDir
    print("   model directory {}".format(modDir))

    modDict = {}
    nullDict = {}
    for contigLengthk in ['0.15', '0.3', '0.5', '1'] :
        modPattern = 'model_siamese_varlen_'+contigLengthk+'k'
        modName = [ x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5") ][0]
        #model_1000 = load_model(os.path.join(modDir, modName))
        modDict[contigLengthk] = load_model(os.path.join(modDir, modName))

        # Read predictions
        Y_pred_file = [ x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x ][0]
        with open(os.path.join(modDir, Y_pred_file)) as f:
            tmp = [line.split() for line in f][0]
            Y_pred = [float(x) for x in tmp ]

        # Read true values
        Y_true_file = [ x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x ][0]
        with open(os.path.join(modDir, Y_true_file)) as f:
            tmp = [ line.split()[0] for line in f]
            Y_true = [ float(x) for x in tmp ]

        nullDict[contigLengthk] =  Y_pred[:Y_true.index(1)]

    #### Step 0: function for predicting viral score using the trained model ####
    def pred(ID) :
        codefw = code[ID]
        codebw = codeR[ID]
        head = seqname[ID]

        #print('predicting '+head)
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

        writef = predF.write('\t'.join([head, str(seqL), str(float(score)), str(float(pvalue))])+'\n')
        flushf = predF.flush()

        return [head, float(score), float(pvalue)]

    outfile = os.path.join(output_dir, os.path.basename(input_fa)+'_gt'+str(min_len)+'bp_dvfpred.txt')
    predF = open(outfile, 'w')
    writef = predF.write('\t'.join(['name', 'len', 'score', 'pvalue'])+'\n')
    predF.close()
    predF = open(outfile, 'a')

    BATCH_MAX = args.batch_size

    print("Counting FASTA sequences")
    with open(input_fa, "r") as faLines:
        n_fasta = sum(1 for l in faLines if l.startswith('>'))

    print("2. Encoding and Predicting Sequences.")
    with open(input_fa, 'r') as faLines :
        code = []
        codeR = []
        seqname = []
        seq_total = 0
        batch_size = 0

        for rec in tqdm.tqdm(SeqIO.parse(faLines, 'fasta'), total=n_fasta, desc='Analysing seqs'):

            head = rec.id
            seq = str(rec.seq)

            if seq.count('N') > 0.3 or len(seq) < min_len or len(seq) > max_len:
                 continue

            batch_size += len(seq)

            codefw = encodeSeq(seq)
            seqR = seq.translate(complement_table)[::-1]
            codebw = encodeSeq(seqR)
            code.append(codefw)
            codeR.append(codebw)
            seqname.append(head)
            if len(seqname) % BATCH_MAX == 0 :
                pool = mp.Pool(core_num)
                head, score, pvalue = zip(*pool.map(pred, range(0, len(code))))
                pool.close()

                # Report number of sequences
                seq_total += len(seqname)
                #print("   processed {} sequences".format(seq_total))
                code = []
                codeR = []
                seqname = []
                batch_size = 0

        # finish off last batch if any
        if len(seqname) > 0 :
            codefw = encodeSeq(seq)
            seqR = seq.translate(complement_table)[::-1]
            codebw = encodeSeq(seqR)
            code.append(codefw)
            codeR.append(codebw)
            seqname.append(head)

            pool = mp.Pool(core_num)
            head, score, pvalue = zip(*pool.map(pred, range(0, len(code))))
            pool.close()

            # Report number of sequences
            seq_total += len(seqname)
            print("   processed {} sequences".format(seq_total))

    predF.close()

    print("3. Done. Thank you for using DeepVirFinder.")
    print("   output in {}".format(outfile))
