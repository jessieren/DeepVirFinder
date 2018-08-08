# DeepVirFinder
Identifying viruses from metagenomic data by deep learning

Authors: Jie Ren, Fengzhu Sun

Maintainer: Jie Ren renj@usc.edu


## Description

DeepVirFinder predicts viral sequences using deep learning method. 
The method has good prediction accuracy for short viral sequences, 
so it can be used to predict sequences from the metagenomic data.

DeepVirFinder significantly improves the prediction accuracy compared to our k-mer based method VirFinder by using convolutional neural networks (CNN).
CNN can automatically learn genomic patterns from the viral and prokaryotic sequences and simultaneously build a predictive model based on the learned genomic patterns. 
The learned patterns are represented in the form of weight matrices of size 4 by k, where k is the length of the pattern. 
This representation is similar to the position weight matrix (PWM), the commonly used representation of biological motifs, 
which are also of size 4 by k and each column specifies the probabilities of having the 4 nucleotides at that position.
When only one type of nucleotide can be chosen at each position with probability 1, the motif degenerates to a k-mer. 
Thus, the CNN is a natural generalization of k-mer based model. 
The more flexible CNN model indeed outperforms the k-mer based model on viral sequence prediction problem.



## Dependencies

DeepVirFinder requires Python 3.6 with the packages of numpy, theano and keras.
We recommand the use [Miniconda](https://conda.io/miniconda.html) to install all dependencies. 
After installing Miniconda, simply run

```
conda install python=3.6 numpy theano keras
```



<!-- Copyright and License Information
-----------------------------------

Copyright (C) 2017 University of Southern California

Authors: Jie Ren, Fengzhu Sun

This program is freely available under the terms of the GNU General Public (version 3) as published by the Free Software Foundation (http://www.gnu.org/licenses/#GPL) for academic use. 

Commercial users should contact Dr. Sun at fsun@usc.edu, copyright at the University of Southern California. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. -->

<!--You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.-->

