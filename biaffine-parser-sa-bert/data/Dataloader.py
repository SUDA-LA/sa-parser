from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable

def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    result = []
    for dep in sentence:
        wordid = vocab.word2id(dep.form)
        extwordid = vocab.extword2id(dep.form)
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        relid = vocab.rel2id(dep.rel)
        word = dep.form
        charid = dep.charid
        senid = dep.senid
        id=dep.id
        result.append([wordid, extwordid, tagid, head, relid, word, charid, id, senid])

    return result



def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    length = len(batch[0])
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b]) > length: length = len(batch[b])

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    positions = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []
    sentences = []
    elmosens=[]
    berts=[]

    b = 0
    for sentence in sentences_numberize(batch, vocab):
        index = 0
        sen=[]
        elmosen=[]
        length = len(sentence)
        lengths.append(length)
        elmosen.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        for dep in sentence:
            words[b, index] = dep[0]
            extwords[b, index] = dep[1]
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            sen.append(dep[5])
            masks[b, index] = 1
            positions[b,index] = index
            index += 1
            if dep[7] == 1:
                startcharid = dep[6]
                berts.append(dep[8])
                '''
                if startcharid == 0:
                    print("the char id is 0:",dep[5])
                    print("the sen is is 0:",dep[8])
                if startcharid == 55:
                    print("the char id is 8",dep[5])
                    print("the sen is is 2:",dep[8])
                if startcharid == 37:
                    print("the char id is 37",dep[5])
                    print("the sen is is 1:",dep[8])
                if startcharid == 83:
                    print("the char id is 83",dep[5])
                    print("the sen is is 2:",dep[8])
                '''
        
        elmosen.append(startcharid)
            
        b += 1
        heads.append(head)
        rels.append(rel)
        sentences.append(sen)
        elmosens.append(elmosen)
        
        #use_cuda=True
        #if use_cuda:
        #    positions=positions.cuda()

    return words, extwords, tags, heads, rels, lengths, masks, positions, sentences,elmosens,berts

def batch_variable_depTree(trees, heads, rels, lengths, vocab):
    for tree, head, rel, length in zip(trees, heads, rels, lengths):
        sentence = []
        for idx in range(length):
            sentence.append(Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx]),tree[idx].charid,tree[idx].senid))
        yield sentence



