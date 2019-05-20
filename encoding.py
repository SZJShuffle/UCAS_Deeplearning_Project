import numpy as np
def test():
    print('hello world')

# encoding sequence to matrix ---------------------------------------------------------
# global CNN encoding---------------------------------------------------

#padding sequence to same length(as the input of global CNN)，该函数只能处理一条序列，当处理多条时，需要放到for循环中
def get_RNA_seq_concolutional_array(seq, motif_len = 4):
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
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

'''
import pre_processing
seq1 = pre_processing.load_graphprot_data('ALKBH5')['seq'][0]
print(get_RNA_seq_concolutional_array(seq1))
'''

##填补序列 (<501就填补) ，用N填补,应该是rdp-24的？
def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


##把sequence直接编码成单通道 4xseq_len的matrix，作为global CNN的输入。
def get_bag_data_1_channel(data, max_len = 501):
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


'''
data  = load_graphprot_data('ALKBH5')
get_bag_data_1_channel(data)
get_bag_data_1_channel(data)[0] ##encoding matrix
get_bag_data_1_channel(data)[1] ##labels


'''



# local CNN encoding---------------------------------------------------

### localCNN的输入需要分割序列成为若干个subseq，就用这个函数
def split_overlap_seq(seq, window_size = 101):
    
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


##序列短于101就填补，用N填补
def padding_sequence_new(seq, max_len = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


'''
seq1 = load_graphprot_data('ALKBH5')['seq'][0]
split_overlap_seq(seq1)

'''



##把切割的subsequence编码成7个通道的matrix，作为local CNN的输入。
def get_bag_data(data, channel = 7, window_size = 101):
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
        num_of_ins = len(bag_subt)   ##num_of_ins的值等于tri_fea的数量
        
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

'''
##下面的数据是ALKBH5蛋白正负样本编码成7个通道以后的结果
import pre_processing
data = pre_processing.load_graphprot_data('ALKBH5')
get_bag_data(data)[0] ##bags：序列编码后的数据
get_bag_data(data)[1] ##labels：正负标签

get_bag_data(data)[0][0] ##第一个sample
get_bag_data(data)[0][0][0] ##第一个sample的第一个subsequence的encode

'''





