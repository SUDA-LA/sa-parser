from driver.Layer import *
from driver.Attention import *
from data.Vocab import *


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


class ParserModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding, max_seq):
        super(ParserModel, self).__init__()
        self.config=config
        self.d_model=config.d_model
        self.attention = MyAttentionEncoder(vocab, config, pretrained_embedding, max_seq)

        self.mlp_arc_dep = NonLinear(
            input_size = config.hidden_size,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size = config.hidden_size,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size+config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, \
                                     vocab.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks, positions,sens,elmosens,berts,elmofile):

        outputs = self.attention(words, extwords, tags, masks, positions, sens,elmosens,berts,elmofile)#outputs=batch_size*sequence_length 6*78*100

        #print(outputs) #outputs's size is 73*6*800
        #print(_) two variables:size are 3*6*800 three layers
        #outputs = outputs.transpose(1, 0)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)#6*73*800

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)
        #print(x_all_dep)#6*73*600
        #print(x_all_head)#6*73*600

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)
        #print(x_all_dep)#6*73*600
        
        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)
        #print(x_arc_dep)#6*73*500

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)#6*73*73*1
        arc_logit = torch.squeeze(arc_logit, dim=3)#6*73*73
        #print(arc_logit)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)#6*73*100
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)
        #print(x_rel_dep)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)#6*73*73*43
        #print(arc_logit.nonzero())
        
        #print(rel_logit_cond.nonzero())
        return arc_logit, rel_logit_cond
