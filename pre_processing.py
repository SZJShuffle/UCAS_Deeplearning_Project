import numpy as np
import os 
# Pre processing:read and load fasta file ,
# and transform a fasta file into the format that pytorch recognized.


def read_seq_graphprot(seq_file, label = 1):
    """
    load fasta file.
    """
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')   ## let T to U
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels




def read_data_file(posifile, negafile = None, train = True):
    """
    save a positive or negative label sample into dict.
    
    """
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data




##path需要改成自己存放数据的目录
def load_graphprot_data(protein, train = True, path = './data'):  
    """
    load data with positive and negative protein sample in dict.

    """
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data
