### to perform downstream analysis
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import argparse

import numpy as np
from pre_processing import *
from encoding import *

from sklearn.metrics import roc_curve, auc, roc_auc_score


def get_all_data(protein, channel = 7):
    data = load_graphprot_data(protein)
    test_data = load_graphprot_data(protein, train = False)
    #pdb.set_trace()
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data)
        test_bags, true_y = get_bag_data_1_channel(test_data)
    else:
        train_bags, label = get_bag_data(data)
        test_bags, true_y = get_bag_data(test_data)

    return train_bags, label, test_bags, true_y


class CNN(nn.Module):
    def __init__(self, nb_filter = 16, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 107, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1   
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1

        print ('out1_size',out1_size)  #98  (107+0-9-1)/1 + 1
        print ('maxpool_size',maxpool_size) #96   ((98+0-2)-1)/1 + 1

        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = (maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool2_size = (out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.drop1 = nn.Dropout(p=0.25)
        print ('out2_size',out2_size)   # 87
        print ('maxpool2_size',maxpool2_size) # 85         

        print ('maxpool2_size * nb_filter', maxpool2_size* nb_filter)  ##1360 (16*85)

        self.fc1 = nn.Linear(int(maxpool2_size*nb_filter), hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def layer1out(self, x):
        x = np.array(x)
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

def loadGlobalCNN(file):
    globalCNN = CNN(channel = 1,labcounts = 4,window_size = 501+6)
    globalCNN.load_state_dict(torch.load(file,map_location='cpu'))    ##if in cpu-only machine
    return globalCNN 

def loadLocalCNN(file):
    localCNN = CNN()
    localCNN.load_state_dict(torch.load(file,map_location='cpu'))     ##if in cpu-only machine
    return localCNN




def plot_filter_logo(filter_outs,seqs,out_prefix,filter_size = 7, raw_t=0, maxpct_t=0.5):
    """
    params:
    ------
    raw_t:
        threshold value
    filter_outs:
        output value of first layer convolution.
    filter_size:
        presetting motif length, default = 7.
    seqs:
        raw sequence data.
    out_prefix:
        prefix of output file.e.g:'./result/filter1_logo'
    """
    
    if maxpct_t:
        all_outs = np.ravel(filter_outs)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_prefix, 'w')
    filter_count = 0
    for i in range(filter_outs.shape[0]): #2410
        for j in range(filter_outs.shape[1]): #498
                if filter_outs[i,j] > raw_t:
                    kmer = seqs[i][j:j+filter_size]
                    incl_kmer = len(kmer) - kmer.count('N')
                    if incl_kmer <filter_size:
                        continue
                    print ( '>%d_%d' % (i,j),file = filter_fasta_out)
                    print (kmer ,file = filter_fasta_out )
                    filter_count += 1
    filter_fasta_out.close()



def make_filter_pwm(filter_fasta):
    """
    Make a PWM for this filter from its top hits 
    """

    nts = {'A':0, 'C':1, 'G':2, 'U':3}
    pwm_counts = []
    nsites = 4 # pseudocounts
    for line in open(filter_fasta):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append(np.array([1.0]*4))

            # count
            for i in range(len(seq)):
                try:
                    pwm_counts[i][nts[seq[i]]] += 1
                except KeyError:
                    pwm_counts[i] += np.array([0.25]*4)

    # normalize
    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j]/float(nsites) for j in range(4)])
    #print( '''Filter %s 's PWM ''' % num,file = pwm_file)
    #print(np.array(pwm_freqs),file = pwm_file)
    #print('number of binding site: %d'% (nsites -4),file = pwm_file)
    #pwm_file.close()
    return np.array(pwm_freqs), nsites-4


def meme_intro(meme_file, seqs):
    ''' 
    Open MEME motif format file and print intro
    ------
    Params:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs
    ------
    Returns:
        mem_out : open MEME file
    '''
    nts = {'A':0, 'C':1, 'G':2, 'U':3}

    # count
    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    # open file for writing
    meme_out = open(meme_file, 'w')

    # print intro material
    print('MEME version 4',file = meme_out)
    print('',file = meme_out)
    print('ALPHABET= ACGU',file = meme_out)
    print('',file = meme_out)
    print('Background letter frequencies:',file = meme_out)
    print('A %.4f C %.4f G %.4f U %.4f' % tuple(nt_freqs),file = meme_out)
    print('',file = meme_out)
    #meme_out.close()
    return meme_out

def meme_add(meme_out, f, filter_pwm, nsites, trim_filters=False):
    ''' 
    Print a filter to the growing MEME file
    ------
    Params:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    '''
    if not trim_filters:
        ic_start = 0
        ic_end = filter_pwm.shape[0]-1
    else:
        ic_t = 0.2

        # trim PWM of uninformative prefix
        ic_start = 0
        while ic_start < filter_pwm.shape[0] and info_content(filter_pwm[ic_start:ic_start+1]) < ic_t:
            ic_start += 1

        # trim PWM of uninformative suffix
        ic_end = filter_pwm.shape[0]-1
        while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end+1]) < ic_t:
            ic_end -= 1

    if ic_start < ic_end:
        print('MOTIF filter%d' % f,file = meme_out)
        print('letter-probability matrix: alength= 4 w= %d nsites= %d' % (ic_end-ic_start+1, nsites),file = meme_out)
        

        for i in range(ic_start, ic_end+1):
            print( '%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]),file = meme_out)
        print('',file = meme_out)


def info_content(pwm, transpose=False, bg_gc=0.415):
    """
    Compute PWM information content.
    
    """
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    bg_pwm = [1-bg_gc, bg_gc, bg_gc, 1-bg_gc]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic



def parser_args(parser):
    parser.add_argument('-protein',type=str,help= 'input the protein you have trained,i.e ALKBH5')

def run(args):
    protein = args.protein

    ##reload networks
    file1 = './'+protein+'_param'+'/globalCNN_'+protein+'.pkl'
    file2 = './'+protein+'_param'+'/localCNN_'+protein+'.pkl'   
    glbCNN = loadGlobalCNN(file1)
    lclCNN = loadLocalCNN(file2)

    ##load date
    train_bags, train_labels,test_bags, true_y = get_all_data(protein, channel = 1)   
    
    ##convolution 
    filter_outs = glbCNN.layer1out(test_bags)[:,:,0,:]   

    ##load sequence
    filelist = os.listdir('./data')
    for file in filelist:
        if protein in file and 'ls' in file:
            if 'posi' in file:
                posi_seq_path = file
            else:
                nega_seq_path = file
    posi_seq_path = './data/'+ posi_seq_path
    nega_seq_path = './data/'+ nega_seq_path 
    data = read_data_file(posi_seq_path,nega_seq_path)
    seqs = data['seq']

    nb_filters = 16

    ##save parameters
    isExist = os.path.exists('./'+protein+'_Motif')
    if not isExist:
        os.mkdir('./'+protein+'_Motif')
    out_dir = './'+protein+'_Motif'

    ##the input of tomtom needs a meme-file.here we generate a meme-file.
    meme_out = meme_intro('%s/filters_meme.txt'%out_dir, seqs)
    filters_ic = []
    for f in range(nb_filters):
        print('Filter:%d '% f  )
        
        '''
        print('filter_outs.shape:')
        print(filter_outs.shape)
        print('filter_outs.shape[0]:')
        print(filter_outs[:,f,:].shape[0])
        print('filter_outs.shape[1]:')
        print(filter_outs[:,f,:].shape[1])
        '''
        
        ##generate binding site and plot weblogo picture
        plot_filter_logo(filter_outs[:,f, :],seqs, '%s/filter%d_logo'%(out_dir,f))

        ##make pwm matrix
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_args(parser)
    args = parser.parse_args()
    print('----args-----')
    print(args)
    run(args)
    print('-----done-----')

