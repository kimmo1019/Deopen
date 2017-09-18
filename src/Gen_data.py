'''
This script is used for generating data for Deopen training with down-sampling strategy.
Usage:
    python Gen_data.py -l 1000 -s 100000 -in <inputfile> -out <outputfile>
'''
import numpy as np
from pyfasta import Fasta
import hickle as hkl
import argparse
import gzip

#transfrom a sequence to one-hot encoding matrix
def seq_to_mat(seq):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':4, 'N':4}
    mat = np.zeros((len(seq),5))  
    for i in range(len(seq)):
        mat[i,encoding_matrix[seq[i]]] = 1
    mat = mat[:,:4]
    return mat

#transform a sequence to K-mer vector (default: K=6)
def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        kspec_vec[index] += 1
    return kspec_vec


#assemble all the features into a dictionary
def get_all_feats(spot,genome,label):
    ret = {}
    ret['spot'] = spot
    ret['seq'] = genome[spot[0]][spot[1]:spot[2]]
    ret['mat'] = seq_to_mat(ret['seq'])
    ret['kmer'] = seq_to_kspec(ret['seq'])
    ret['y'] = label
    return ret


#save the preprocessed dataset in hkl format 
def  save_dataset(origin_dataset,save_dir):
    dataset = {}
    for key in origin_dataset[0].keys():
        dataset[key] = [item[key] for item in origin_dataset]
    dataset['seq'] = [item.encode('ascii','ignore') for item in dataset['seq']]
    for key in origin_dataset[0].keys():
        dataset[key] = np.array(dataset[key])
    print "Start to save dataset ..."
    hkl.dump(dataset,save_dir)    
    print 'Dataset generation is finished!'    


#generate dataset
def  generate_dataset(input_file,sample_length,dataset_size,ratio = 1):
    chrom_iter = 0 
    dataset=[]
    chrom_dict=[str(item) for item in range(1,23)]+['X','Y']
    DATA_PATH = './openness_pre/data'
    genome = Fasta(DATA_PATH+'/genome.fa')
    chromosome_len_file=DATA_PATH+'/chromosome.txt'
    with open(chromosome_len_file,'r') as f:
        chrom_lens = f.readlines() 
    openness_records=[]
    with gzip.open(input_file,'r') as f:
        for line in f:
            line = line.split()
            openness_records.append([line[0], int(line[1]), int(line[2]), int(line[4]), float(line[5])/float(line[4])])  

    while(len(dataset)<dataset_size):
        chrom = 'chr'+chrom_dict[chrom_iter]
        records_in_chrom = [item for item in openness_records if item[0]==chrom]        
        samples_pos = [[item[0], (item[1]+item[2])/2 - sample_length/2, (item[1]+item[2])/2 + sample_length/2] for item in records_in_chrom]        
        for each in samples_pos:
            dataset.append(get_all_feats(each,genome,1))
        chrom_len = int(chrom_lens[chrom_iter].strip().split()[-1])
        obin = np.zeros((chrom_len,1))
        for item in records_in_chrom:
            obin[item[1]:item[2]] = 1
        samples_neg = []
        while len(samples_neg) < int(len(records_in_chrom)*ratio):
            middle = np.random.randint(chrom_len)
            start = middle - sample_length/2
            end = middle + sample_length/2
            if np.sum(obin[start:end]) == 0     \
                    and  'N' not in genome[chrom][start:end]    \
                    and  'n' not in genome[chrom][start : end] :
                samples_neg.append([chrom,start,end])
        for each in samples_neg:
            dataset.append(get_all_feats(each,genome,0))
        chrom_iter +=1
    return  dataset
        
        
if  __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Deopen data generation') 
    parser.add_argument('-l', dest='length', type=int, default=1000, help='sequence length')
    parser.add_argument('-s', dest='size', type=int, default=100000, help='number of samples')
    parser.add_argument('-in', dest='input', type=str, help='file of raw input data')
    parser.add_argument('-out', dest='output', type=str, help='output file')
    args = parser.parse_args()   
    dataset = generate_dataset(args.input,args.length,args.size)
    save_dataset(dataset,args.output)
 
    
