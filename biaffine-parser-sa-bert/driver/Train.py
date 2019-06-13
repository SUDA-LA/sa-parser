import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Parser import *
from data.Dataloader import *
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(data, dev_data,test_data, parser, vocab, config,trainbert,devbert,testbert):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.model.parameters()), config)

    global_step = 0
    best_UAS = 0
    olditer=0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, tags, heads, rels, lengths, masks, positions,sens,elmosens,berts = \
                batch_data_variable(onebatch, vocab)
            parser.model.train()

            #elmofile="./conll/elmo_average/train.1024.bin"
            #elmofile="../../Baidu_PTB_SD/elmo_emb_average/train.1024.bin"
            parser.forward(words, extwords, tags, masks, positions,sens,elmosens,berts,trainbert)
            loss = parser.compute_loss(heads, rels, lengths)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            arc_correct, label_correct, total_arcs = parser.compute_accuracy(heads, rels)
            overall_arc_correct += arc_correct
            overall_label_correct += label_correct
            overall_total_arcs += total_arcs
            uas = float(overall_arc_correct) * 100.0 / overall_total_arcs
            las = float(overall_label_correct) * 100.0 / overall_total_arcs
            during_time = float(time.time() - start_time)
            print("Step:%d, ARC:%.2f, REL:%.2f, Iter:%d, batch:%d, length:%d,time:%.2f, loss:%.2f" \
                %(global_step, float(uas), float(las), iter, batch_iter, overall_total_arcs, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                parser.model.zero_grad()       
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(dev_data, parser, vocab, './devfile/dev.out',devbert)
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                if dev_uas > best_UAS:
                    olditer=iter
                    print("Exceed best uas: history = %.2f, current = %.2f,, Iter:%d" %(best_UAS, dev_uas, iter))
                    arc_cor,rel_cor,arc_tot,test_uas,test_las=evaluate(test_data,parser,vocab,'./testfile/test.'+str(global_step),testbert)
                    print("Test:uas = %d/%d = %.2f, las = %d/%d =%.2f"\
                            %(arc_cor,arc_tot,test_uas,rel_cor,arc_tot,test_las))
                    best_UAS = dev_uas
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(parser.model.state_dict(), config.save_model_path )
        if iter-olditer>151:
            print("model is saved as the decay iter is longger than 100")
            break


def evaluate(data, parser, vocab, outputFile,elmofile):
    start = time.time()
    parser.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, tags, heads, rels, lengths, masks, positions,sens,elmosens,berts = \
            batch_data_variable(onebatch, vocab)
        count = 0
        arcs_batch, rels_batch = parser.parse(words, extwords, tags, lengths, masks, positions,sens, elmosens,berts,elmofile)
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon,weight_decay=1e-5)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda' ,action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab,max_seq = creatVocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = ParserModel(vocab, config, vec, max_seq)
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    parser = BiaffineParser(model, vocab.ROOT)

    data = read_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    tbert=open('/data1/yli/paser/attention-biaffine-parser/deal-bert/data/PTB/ptb-bert/train-raw.last','rb')
    dbert=open('/data1/yli/paser/attention-biaffine-parser/deal-bert/data/PTB/ptb-bert/dev-raw.last','rb')
    tebert=open('/data1/yli/paser/attention-biaffine-parser/deal-bert/data/PTB/ptb-bert/test-raw.last','rb')
    trainbert=pickle.load(tbert)
    devbert=pickle.load(dbert)
    testbert=pickle.load(tebert)

    print("train len:",len(trainbert))
    print("dev len:",len(devbert))
    print("test len:",len(testbert))

    train(data, dev_data, test_data, parser, vocab, config,trainbert,devbert,testbert)
