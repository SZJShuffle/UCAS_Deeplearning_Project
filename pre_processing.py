import numpy as np
import os 
# pre processing:read and load fasta file  -----------------------------------------
# 读取数据的目的是最终把数据转换成pytorch能够识别的input格式

##读取fasta序列
def read_seq_graphprot(seq_file, label = 1):
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



##把正或负sample存到一个data字典里
def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data


'''
##测试read_seq_graphprot 和 read_data_file
nega_seq_path = 'D:/R-exercise/ideepeData/ALKBH5_Baltz2012.ls.negatives.fa'
posi_seq_path = 'D:/R-exercise/ideepeData/ALKBH5_Baltz2012.ls.positives.fa'
nega_seq_path = 'D:/R-exercise/ideepeData/ALKBH5_Baltz2012.train.negatives.fa'
posi_seq_path = 'D:/R-exercise/ideepeData/ALKBH5_Baltz2012.train.positives.fa'

nega_seq = read_seq_graphprot(nega_seq_path,label = 0)
posi_seq = read_seq_graphprot(posi_seq_path,label = 1)

len(read_data_file(posi_seq_path)['Y'])
len(read_data_file(nega_seq_path)['Y'])


read_data_file(posi_seq_path,nega_seq_path)['Y']
len(read_data_file(posi_seq_path,nega_seq_path)['Y'])



'''

##加载数据(正和负的protein sample都放到data字典里)
##load_graphprot_data返回的序列并没有进行编码。

##path需要改成自己存放数据的目录
def load_graphprot_data(protein, train = True, path = './data'):  
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

'''
##测试load_graphprot_data
load_graphprot_data('ALKBH5')['seq']
load_graphprot_data('ALKBH5')['Y']

load_graphprot_data('CAPRIN1',train = false)

len(load_graphprot_data('ALKBH5')['seq'])  ##2410 = 1197 + 1213
len(load_graphprot_data('ALKBH5')['Y'])

'''






