import numpy as np

######### encoding sequence to matrix ###############
### global CNN encoding

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    """
    padding sequence to same length(as the input of global CNN).
    """
    seq = seq.replace('U', 'T')  ## let U to T
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array




def padding_sequence(seq, max_len = 501, repkey = 'N'):
    """
    padding sequence (if less than 501) with 'N'.
    """
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq



def get_bag_data_1_channel(data, max_len = 501):
    """
    enconding sequence into single channel( 4*sequence_length matrix) as the input of GlobalCNN.
    """
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        #bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)        
        bags.append(np.array(bag_subt))        
    return bags, labels



## local CNN encoding


def split_overlap_seq(seq, window_size = 101):
    """
    split the input of localCNN, into multiple subsequence.
    """
    overlap_size = 20
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            #pdb.set_trace()
            #start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len = window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs


def padding_sequence_new(seq, max_len = 101, repkey = 'N'):
    """
    padding sequence with N if less than 101(default).
    """
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


def get_bag_data(data, channel = 7, window_size = 101):
    """
    encoding splited subsequence(by function: split_overlap_seq) as 7 channels matrix, as the input of localCNN.
    """
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        #flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)   
        num_of_ins = len(bag_subt)   ## num_of_ins equals tri_fea
        
        if num_of_ins >channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                #bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        
        bags.append(np.array(bag_subt))
        
    return bags, labels
