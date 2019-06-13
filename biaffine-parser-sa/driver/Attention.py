import torch
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd
import numpy as np
from driver.Model import *

def reset_bias_with_orthogonal(bias):
    bias_temp = torch.nn.Parameter(torch.FloatTensor(bias.size()[0], 1))
    nn.init.orthogonal(bias_temp)
    bias_temp = bias_temp.view(-1)
    bias.data = bias_temp.data

def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)#6*98 ;0.67
    #print(word_masks)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)#6*78 ;0,1
    #print(word_masks)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)#batch_size*seq_length 1,1.5,3
    #print(scale)
    word_masks *= scale#batch_size*seq_length 0,1,1.5
    tag_masks *= scale
    #print(word_masks)
    word_masks = word_masks.unsqueeze(dim=2)#6*78*1
    #print(word_masks)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    return word_embeddings, tag_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
        seq_length, batch_size, hidden_size = inputs.size()
        drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
        drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
        drop_masks = drop_masks / (1 - dropout)
        drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
        inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class MyAttentionEncoder(nn.Module):
    def __init__(self,vocab,config,pretrained_embedding,
            n_max_seq,n_layers=3,n_head=8,d_k=25,d_v=25,d_word_vec=200,d_model=200,d_inner_hid=800,dropout=0.2):
        super(MyAttentionEncoder,self).__init__()

        n_position=n_max_seq+1
        #self.n_max_seq=n_max_seq
        self.n_max_seq=1000
        self.d_model=d_model
        self.config=config
        self.position_enc=nn.Embedding(n_position,config.word_dims+config.tag_dims,padding_idx=0)
        self.position_enc.weight.data=position_encoding_init(n_position,config.word_dims+config.tag_dims)

        self.word_emb=nn.Embedding(vocab.vocab_size,config.word_dims,padding_idx=0)
        self.extword_emb=nn.Embedding(vocab.extvocab_size,config.word_dims,padding_idx=0)
        self.tag_emb=nn.Embedding(vocab.tag_size,config.tag_dims,padding_idx=0)
        #word_init = np.random.randn(vocab.vocab_size,config.word_dims).astype(np.float32)
        #word_init = np.zeros((vocab.vocab_size,config.word_dims),dtype=np.float32)
        #self.word_emb.weight.data.copy_(torch.from_numpy(word_init))
        nn.init.normal(self.word_emb.weight, 0.0, 1.0 / (200 ** 0.5))

        self.bias = nn.Parameter(torch.ones(200), requires_grad=True)
        nn.init.normal(self.bias, 0.0, 1.0 / (200 ** 0.5))

        tag_init = np.random.randn(vocab.tag_size,config.tag_dims).astype(np.float32)
        self.tag_emb.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_emb.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_emb.weight.requires_grad=False

        self.dropout=nn.Dropout(config.dropout_emb)


        self.layer_stack=nn.ModuleList([Encoderlayer(d_model,d_inner_hid,n_head,d_k,d_v,dropout=dropout) for _ in
            range(config.n_layer)])

        #print("init is end")

    def forward(self,word_seq,extwords,tags,masks,pos_seq,return_attn=False):
        enc_input=self.word_emb(word_seq)#mb_size*n_max_seq*d_word_vec
        extenc_input=self.extword_emb(extwords)#mb_size*n_max_seq*d_word_vec
        enc_input=enc_input + extenc_input#mb_size*n_max_seq*d_word_vec
        tag_input=self.tag_emb(tags)#mb_size*n_max_seq*d_word_vec
        position_input = self.position_enc(pos_seq)#mb_size*n_max_seq*d_word_vec
        if self.training:
            #enc_input = self.dropout(enc_input)
            #position_input = self.dropout(position_input)
            #position_input = drop_sequence_sharedmask(position_input,0.33)
            enc_input, tag_input = drop_input_independent(enc_input, tag_input, self.config.dropout_emb)
        enc_input =torch.cat((enc_input,tag_input),dim=2)
        enc_input = enc_input+self.bias
        enc_input += position_input#mb_size*n_max_seq*d_word_vec
        if self.training:
            enc_input = drop_sequence_sharedmask(enc_input,0.33)

        if return_attn:
            enc_slf_attns=[]
        
        enc_output=enc_input
        enc_slf_attn_mask=get_attn_padding_mask(word_seq,word_seq)#it can be replaced by masks
        #print(enc_slf_attn_mask.size())
        #print("get mask end")
        for enc_layer in self.layer_stack:
            enc_output,enc_slf_attn = enc_layer(enc_output,slf_attn_mask=enc_slf_attn_mask)
            if return_attn:
                enc_slf_attns += enc_slf_attn
        if return_attn:
            return enc_output,enc_slf_attns
        else:
            return enc_output

def get_attn_padding_mask(seq_q,seq_k):
    mb_size,len_q=seq_q.size()
    mb_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)#mb_size*1*len_k
    pad_attn_mask=pad_attn_mask.expand(mb_size,len_q,len_k)#mb_size*len_q*len_k
    return pad_attn_mask



def position_encoding_init(n_position,d_pos_vec):
    position_enc=np.array([[pos/np.power(10000,2*(j//2)/d_pos_vec) for j in range(d_pos_vec)] if pos!=0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    position_enc[1:,0::2] = np.sin(position_enc[1:,0::2]) #dim=2i
    position_enc[1:,1::2] = np.cos(position_enc[1:,1::2]) #dim =2i+1
    position_enc=torch.from_numpy(position_enc).type(torch.FloatTensor)
    return position_enc

class Encoderlayer(nn.Module):
    
    def __init__(self,d_model,d_inner_hid,n_head,d_k,d_v,dropout=0.1):
        super(Encoderlayer,self).__init__()
        self.slf_attn=MultiHeadAttention(d_model,n_head,d_k,d_v,dropout=dropout)
        self.pos_ffn=PositionWiseFeedForward(d_model,d_inner_hid,dropout=dropout)

    def forward(self,enc_input,slf_attn_mask=None):
        first_feedforward=True
        if first_feedforward:
            enc_output=self.pos_ffn(enc_input)
            enc_output,enc_slf_attn=self.slf_attn(enc_output,enc_output,enc_output,attn_mask=slf_attn_mask)
        else:
            enc_output,enc_slf_attn=self.slf_attn(enc_input,enc_input,enc_input,attn_mask=slf_attn_mask)
            enc_output=self.pos_ffn(enc_output)
        return enc_output,enc_slf_attn

class Linear(nn.Module):
    def __init__(self,d_in,d_out,bias=True):
        super(Linear,self).__init__()
        self.linear = nn.Linear(d_in,d_out,bias=bias)
        #init.xavier_normal(self.linear.weight)
        nn.init.orthogonal(self.linear.weight)
        if bias:
            reset_bias_with_orthogonal(self.linear.bias)

    def forward(self,x):
        return self.linear(x)

class Bottle(nn.Module):
    def forward(self,input):
        if len(input.size())<=2:
            return super(Bottle,self).forward(input)
        size=input.size()[:2]
        out=super(Bottle,self).forward(input.view(size[0]*size[1],-1))
        return out.view(size[0],size[1],-1)

class BottleSoftmax(Bottle,nn.Softmax):
    pass

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.temper=np.power(d_model,0.5)
        self.dropout=nn.Dropout(attn_dropout)
        self.softmax=nn.Softmax(dim=2)
        #self.softmax=BottleSoftmax()

    def forward(self,q,k,v,attn_mask=None):
        attn = torch.bmm(q,k.transpose(1,2))/self.temper
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask,-float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output=torch.bmm(attn,v)
        return output,attn

class LayerNormalization(nn.Module):

    def __init__(self,d_hid,eps=1e-3):
        super(LayerNormalization,self).__init__()
        self.eps=eps
        self.a2=nn.Parameter(torch.ones(d_hid),requires_grad=True)
        self.b2=nn.Parameter(torch.zeros(d_hid),requires_grad=True)
    
    def forward(self,z):
        if(z.size(1)==1):
            return z
        mu=torch.mean(z,keepdim=True,dim=-1)
        sigma=torch.std(z,keepdim=True,dim=-1)
        ln_out=(z-mu.expand_as(z))/(sigma.expand_as(z)+self.eps)
        ln_out=ln_out*self.a2.expand_as(ln_out)+self.b2.expand_as(ln_out)
        return ln_out

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model,n_head,d_k,d_v,dropout=0.2):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head
        self.d_v=d_v
        self.d_k=d_k

        self.w_qs=nn.Parameter(torch.FloatTensor(n_head,d_model,d_k))
        self.w_ks=nn.Parameter(torch.FloatTensor(n_head,d_model,d_k))
        self.w_vs=nn.Parameter(torch.FloatTensor(n_head,d_model,d_v))
        
        self.attention=ScaledDotProductAttention(d_model)
        self.linear=Linear(n_head*d_v,d_model,bias=False) #xiugai
        self.layer_norm=LayerNormalization(d_model)

        #self.dropout=nn.Dropout(dropout)

        #init.xavier_normal(self.w_qs)
        #init.xavier_normal(self.w_ks)
        #init.xavier_normal(self.w_vs)
        nn.init.orthogonal(self.w_qs)
        nn.init.orthogonal(self.w_ks)
        nn.init.orthogonal(self.w_vs)

    def forward(self,q,k,v,attn_mask=None):
        d_k,d_v=self.d_k,self.d_v
        n_head=self.n_head
        residual=q

        mb_size,len_q,d_model=q.size()
        mb_size,len_k,d_model=k.size()
        mb_size,len_v,d_model=v.size()
        
        q_s=q.repeat(n_head,1,1).view(n_head,-1,d_model)
        k_s=k.repeat(n_head,1,1).view(n_head,-1,d_model)
        v_s=v.repeat(n_head,1,1).view(n_head,-1,d_model)

        q_s=torch.bmm(q_s,self.w_qs).view(-1,len_q,d_k)
        k_s=torch.bmm(k_s,self.w_ks).view(-1,len_k,d_k)
        v_s=torch.bmm(v_s,self.w_vs).view(-1,len_v,d_v)

        outputs,attns=self.attention(q_s,k_s,v_s,attn_mask=attn_mask.repeat(n_head,1,1))
        outputs=torch.cat(torch.split(outputs,mb_size,dim=0),dim=-1)
        outputs=self.linear(outputs)
        #outputs=self.dropout(outputs)
        if self.training:
            outputs=drop_sequence_sharedmask(outputs,0.2)
        outputs=self.layer_norm(outputs+residual)
        
        #outputs=outputs+residual
        return outputs,attns

class PositionWiseFeedForward(nn.Module):

    def __init__(self,d_hid,d_inner_hid,dropout=0.2):
        super(PositionWiseFeedForward,self).__init__()
        #self.w1=nn.Conv1d(d_hid,d_inner_hid,1)
        #self.w2=nn.Conv1d(d_inner_hid,d_hid,1)
        self.layer_norm=LayerNormalization(d_hid)
        #self.dropout=nn.Dropout(dropout)#0.2, old is 0.1 
        #self.relu_dropout=nn.Dropout(0.1)#zengjia 0.1
        self.relu=nn.ReLU()
        use_linear=True
        if use_linear:
            self.linear1=Linear(d_hid,d_inner_hid)
            self.linear2=Linear(d_inner_hid,d_hid)
            
    def forward(self,x):
        residual=x
        use_linear=True
        if use_linear:
            output=self.relu(self.linear1(x))
            if self.training:
                output=drop_sequence_sharedmask(output,0.1)
            #output=self.relu_dropout(output)
            output=self.linear2(output)
        else:
            output=self.relu(self.w1(x.transpose(1,2)))
            output=self.w2(output).transpose(2,1)
       # output=self.dropout(output)
        if self.training:
            output=drop_sequence_sharedmask(output,0.2)
        output=self.layer_norm(output+residual)
        #output=output+residual
        return output
        


if __name__=='__main__':
   #net=ScaledDotProductAttention(512)
   # net=MultiHeadAttention(8,8,4,4)
    #forwd=PositionWiseFeedForward(8,16)
    #net=Encoderlayer(8,16,8,4,4)
    #params=list(net.parameters())
    #print(len(params))
    #x=autograd.Variable(torch.randn(6,7,8))
    #print(x)
   # output,attn=net(x)
   # print(output)
    
    seq=[[1,3,3,4,0,0],[1,2,3,4,5,6],[1,2,3,0,0,0],[1,2,0,4,5,0]]
    seq=np.array(seq)
    seq=torch.from_numpy(seq)
    seq=autograd.Variable(seq)
    
    #pad_attn=get_attn_padding_mask(seq,seq)
    #print(attn)
    net2=MyAttentionEncoder(10,6)
    params=list(net2.parameters())
    print(len(params))

    output=net2(seq,seq)
    print(output)
    
    #output=forwd(output)
    #print(output)









