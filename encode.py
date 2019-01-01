#!/usr/bin/env python
# title             :encode.py
# description       :Fragment genomes into fixed length sequences, and one-hot encode it into matrix.
# author            :Jie Ren renj@usc.edu
# date              :20180807
# version           :1.0
# usage             :python encode.py -i ./train_example/val/host_val.fa -l 1000 -p host
# required packages :numpy, theano, keras, scikit-learn, Biopython
# conda create -n dvf python=3.6 numpy theano keras scikit-learn Biopython
#==============================================================================

import os, sys
from Bio.Seq import Seq
import numpy as np
import optparse
    
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
    

prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-i", "--fileName", action = "store", type = "string", dest = "fileName",
									help = "fileName")
parser.add_option("-l", "--contigLength", action = "store", type = int, dest = "contigLength",
									help = "contigLength")
parser.add_option("-p", "--contigType", action = "store", type = "string", dest = "contigType",
									help = "contigType, virus or host")
(options, args) = parser.parse_args()
if (options.fileName is None or options.contigLength is None ) :
	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
	parser.print_help()
	sys.exit(0)

contigType = options.contigType
contigLength = options.contigLength
contigLengthk = contigLength/1000
if contigLengthk.is_integer() :
    contigLengthk = int(contigLengthk)


fileName = options.fileName
NCBIName = os.path.splitext((os.path.basename(fileName)))[0]

fileDir = os.path.dirname(fileName)
outDir0 = fileDir
outDir = os.path.join(outDir0, "encode")
if not os.path.exists(outDir):
    os.makedirs(outDir)

fileCount = 0
with open(fileName, 'r') as faLines :
    code = []
    codeR = []
    seqname = []
    head = ''
    lineNum = 0
    seqCat = ''
    flag = 0
    for line in faLines :
        if flag == 0 and line[0] == '>' :
            lineNum += 1
            head = line.strip()
            continue
        elif line[0] != '>' :
            seqCat = seqCat + line.strip()
            flag += 1
            lineNum += 1
        elif flag > 0 and line[0] == '>' :
            lineNum += 1
#            print("seqCatLen="+str(len(seqCat)))
            pos = 0
            posEnd = pos + contigLength
            while posEnd <= len(seqCat) :
                contigName = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k#"+head.split('/')[-1]+"#"+str(pos)+"#"+str(posEnd)
                seq = seqCat[pos:posEnd]

                countN = seq.count("N")
                if countN/len(seq) <= 0.3 : 
                    seqname.append(">"+contigName)
                    seqname.append(seq)
                    seq_code = encodeSeq(seq)
                    code.append(seq_code)
                    seqR = Seq(seq).reverse_complement()
                    seqR_code = encodeSeq(seqR)
                    codeR.append(seqR_code)
#                else :
#                    print("remove seq for >30% Ns {}".format(seq))

                pos = posEnd
                posEnd = pos + contigLength

                if len(seqname) > 0 and len(seqname) % 4000000 == 0 :
                    print("lineNum="+str(lineNum)+",contigNum="+str(len(seqname)))
                    fileCount += 1
                    codeFileNamefw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount)+"_seq"+str(len(code))+"_codefw.npy"
                    codeFileNamebw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount)+"_seq"+str(len(codeR))+"_codebw.npy"
                    nameFileName = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount)+"_seq"+str(int(len(seqname)/2))+".fasta"     
                    print("encoded sequences are saved in {}".format(codeFileNamefw))
                    np.save( os.path.join(outDir, codeFileNamefw), np.array(code) )
                    np.save( os.path.join(outDir, codeFileNamebw), np.array(codeR) )
                    seqnameF = open(os.path.join(outDir, nameFileName), "w")
                    seqnameF.write('\n'.join(seqname) + '\n')
                    seqnameF.close()

                    code = []
                    codeR = []
                    seqname = []

            flag = 0
            seqCat = ''
            head = line.strip()
          
    if flag > 0 :
        lineNum += 1
#        print("seqCatLen="+str(len(seqCat)))
        pos = 0
        posEnd = pos + contigLength
        while posEnd <= len(seqCat) :
            contigName = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k#"+head.split('/')[-1]+"#"+str(pos)+"#"+str(posEnd)
            seq = seqCat[pos:posEnd]

            countN = seq.count("N")
            if countN/len(seq) <= 0.3 : 
                seqname.append(">"+contigName)
                seqname.append(seq)
                seq_code = encodeSeq(seq)
                code.append(seq_code)
                seqR = Seq(seq).reverse_complement()
                seqR_code = encodeSeq(seqR)
                codeR.append(seqR_code)
#            else :
#                print("remove seq for >30% Ns {}".format(seq))

            pos = posEnd
            posEnd = pos + contigLength

            if len(seqname) > 0 and len(seqname) % 4000000 == 0 :
                print("lineNum="+str(lineNum)+",contigNum="+str(len(seqname)))
                fileCount += 1
                codeFileNamefw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount)+"_seq"+str(len(code))+"_codefw.npy"
                codeFileNamebw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount)+"_seq"+str(len(codeR))+"_codebw.npy"
                nameFileName = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount)+"_seq"+str(int(len(seqname)/2))+".fasta"
                print("encoded sequences are saved in {}".format(codeFileNamefw))
                np.save( os.path.join(outDir, codeFileNamefw), np.array(code) )
                np.save( os.path.join(outDir, codeFileNamebw), np.array(codeR) )
                seqnameF = open(os.path.join(outDir, nameFileName), "w")
                seqnameF.write('\n'.join(seqname) + '\n')
                seqnameF.close()

                code = []
                codeR = []
                seqname = []

if len(code) > 0 :
    codeFileNamefw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount+1)+"_seq"+str(len(code))+"_codefw.npy"
    codeFileNamebw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount+1)+"_seq"+str(len(codeR))+"_codebw.npy"
    nameFileName = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_num"+str(fileCount+1)+"_seq"+str(int(len(seqname)/2))+".fasta"
    print("encoded sequences are saved in {}".format(codeFileNamefw))
    np.save( os.path.join(outDir, codeFileNamefw), np.array(code) )
    np.save( os.path.join(outDir, codeFileNamebw), np.array(codeR) )
    seqnameF = open(os.path.join(outDir, nameFileName), "w")
    seqnameF.write('\n'.join(seqname) + '\n')
    seqnameF.close()

           
