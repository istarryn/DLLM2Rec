import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from SASRecModules_ori import *
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")
    # setting
    parser.add_argument('--model_name', type=str, default='SASRec', help='model name.')
    parser.add_argument('--data', nargs='?', default='game',  help='movie, game, toy')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ')
    parser.add_argument('--cuda', type=int, default=3, help='cuda device.')                    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')        
    parser.add_argument('--epoch', type=int, default=1000, help='Number of max epochs.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_negtive_items', type=int, default=1, help='neg sample')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_decay', type=float, default=0, help='weight decay')
    # caser
    parser.add_argument('--num_filters', type=int, default=16, help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]', help='Specify the filter_size')
    # dro
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for dros loss')
    parser.add_argument('--beta', type=float, default=1.0, help='for robust radius')
    # dllm2rec
    parser.add_argument('--ed_weight', type=float, default=0.2, help='weight for collaborative embedding distillation')
    parser.add_argument('--lam', type=float, default=0.8, help='weight for importance-aware ranking distillation') 
    parser.add_argument('--candidate_topk', type=int, default=10, help='top k items from llm')
    parser.add_argument('--gamma_position', type=float, default=0.3, help='weight for ranking position-aware')
    parser.add_argument('--gamma_confidence', type=float, default=0.5, help='weight for ranking importance-aware')
    parser.add_argument('--gamma_consistency', type=float, default=0.1, help='weight for ranking consistency-aware')
    parser.add_argument('--beta2', type=float, default=1.0, help='weight for importance-aware ranking distillation')   
    return parser.parse_args()

    
class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

        self.fc_4096_64 = nn.Linear(4096, 64)

    def forward(self, states, len_states, llm_emb=None):
        # Supervised Head
        emb = self.item_embeddings(states)
        if llm_emb != None:
            llm_emb = self.fc_4096_64(llm_emb)
            emb = emb + args.ed_weight * llm_emb
        if 0 in len_states:
            len_states = [max(1, length) for length in len_states]
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc_4096_64 = nn.Linear(4096, 64)

    def forward(self, states, len_states, llm_emb=None):
        input_emb = self.item_embeddings(states)
        if llm_emb != None:
            llm_emb = self.fc_4096_64(llm_emb)
            input_emb = input_emb + args.ed_weight * llm_emb
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
           
        self.fc_4096_64 = nn.Linear(4096, 64)
        self.relu = nn.ReLU()

    def forward(self, states, len_states, llm_emb=None):
        inputs_emb = self.item_embeddings(states)
        if llm_emb != None:
            llm_emb = self.fc_4096_64(llm_emb)
            inputs_emb = inputs_emb + args.ed_weight * llm_emb
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)        
        seq *= mask
        seq_normalized = self.ln_1(seq)

        mh_attn_out = self.mh_attn(seq_normalized, seq) # cost time

        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)  
        state_hidden = extract_axis_1(ff_out, len_states - 1) 
        supervised_output = self.s_fc(state_hidden).squeeze()

        return supervised_output
    

def myevaluate(model, test_data, device, llm_all_emb=None):
    states = []
    len_states = []
    actions = []
    total_purchase = 0
    import csv
    with open(test_data, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            seq = eval(row['seq'])
            len_seq = int(row['len_seq'])
            next_item = float(row['next'])
            states.append(seq)
            len_states.append(len_seq)
            actions.append(next_item)
            total_purchase += 1

    states = np.array(states)
    states = states.astype(np.int64)
    states = torch.LongTensor(states)
    states = states.to(device)

    if llm_all_emb != None:
        seq = states
        llm_emb = torch.zeros(seq.size(0), seq.size(1), 4096, device=device)
        mask = seq < llm_all_emb.size(0)
        llm_emb[mask] = llm_all_emb[seq[mask]]
        llm_emb = llm_emb.to(device)
    else:
        llm_emb = None

    # model.forward
    prediction = model.forward(states, np.array(len_states),llm_emb) # [num_test,num_item]
    sorted_list = torch.argsort(prediction.detach()).cpu().numpy()

    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    calculate_hit(sorted_list=sorted_list, topk=topk, true_items=actions, hit_purchase=hit_purchase,
                  ndcg_purchase=ndcg_purchase)

    print('#' * 120)
    hr_list = []
    ndcg_list = []
    print('hr@{}\tndcg@{}\thr@{}\tndcg@{}\thr@{}\tndcg@{}\thr@{}\tndcg@{}'.format(topk[0], topk[0], topk[1], topk[1],
                                                                                  topk[2], topk[2], topk[3], topk[3]))
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / total_purchase
        ng_purchase = ndcg_purchase[i] / total_purchase
        hr_list.append(hr_purchase)
        if ng_purchase == 0.0:
            ndcg_list.append(ng_purchase)
        else:
            ndcg_list.append(ng_purchase[0, 0])
        if i == 3:
            hr_20 = hr_purchase
            ndcg_20 = ng_purchase
            rec_list = sorted_list[:, -topk[i]:]

    print(
        '{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1],
                                                                                (ndcg_list[1]), hr_list[2],
                                                                                (ndcg_list[2]), hr_list[3],
                                                                                (ndcg_list[3])))
    print('#' * 120)
    return prediction, hr_list, ndcg_list

def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps

def set_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    s = time.time()
    set_seed(2024)

    args = parse_args()
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items

    topk = [1, 5, 10, 20]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name
    if model_name == "GRU":           
        model = GRU(args.hidden_factor, item_num, seq_size)
    elif model_name == "SASRec":
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    elif model_name == "Caser":
        model = Caser(args.hidden_factor,item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)
    else:
        print("check model name!")
        exit(-1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()
    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    if args.alpha == 0:
        print('without using dros!')
    else:
        ps = calcu_propensity_score(train_data)
        ps = torch.tensor(ps)
        ps = ps.to(device)

    tocf_data_directory = './tocf/' + args.data
    if args.lam != 0:       
        candidate_path = os.path.join(tocf_data_directory, 'myrank_train.txt')
        all_candidate = np.loadtxt(candidate_path)
        all_candidate = torch.LongTensor(all_candidate).to(device) # [train_data_num, k]
        llm_confidence_path = os.path.join(tocf_data_directory, 'confidence_train.txt')
        llm_confidence = np.loadtxt(llm_confidence_path)
        llm_confidence = torch.tensor(llm_confidence, dtype=torch.float).to(device) # [train_data_num, k]
    else:
        print('without using importance-aware ranking distillation!')
    if args.ed_weight != 0:
        llm_all_emb_path = os.path.join(tocf_data_directory, 'all_embeddings.pt')
        llm_all_emb = torch.load(llm_all_emb_path) # [num_item, 4096]
        llm_all_emb = llm_all_emb.to(device)
    else:
        llm_all_emb = None
        print('without using collaborative embedding distillation!')

    total_step = 0
    best_ndcg20 = 0
    best_hr20 = 0
    best_step = 0
    patient = 0
    best_hr_list_result = []
    best_ndcg_list_result = []
    best_accuracy = {}
    best_prediction = 0
    num_rows = train_data.shape[0]
    num_batches = int(num_rows / args.batch_size)

    for i in range(args.epoch):
        s_epoch = time.time()
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size)
            sample = batch.index
            batch = batch.to_dict()
            
            seq = list(batch['seq'].values()) # [batch_size, 10]
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())

            optimizer.zero_grad()

            import ast
            if type(seq[0]) == str:
                seq = [[int(num) for num in ast.literal_eval(s)] for s in seq]
            if type(target[0]) == str:
                target = [ast.literal_eval(s) for s in target]
            target = [int(s) for s in target]
            seq = torch.LongTensor(seq).to(device)
            if model_name == "SASRec":
                len_seq = torch.LongTensor(len_seq).to(device)
            target = torch.LongTensor(target).to(device)
            
            # negtive item sampling 
            real_batch_size = args.batch_size
            num_negtive_items = args.num_negtive_items
            zeros_tensor = torch.zeros((real_batch_size, item_num+1), device=device)
            zeros_tensor[torch.arange(real_batch_size).unsqueeze(1).repeat(1, 10), seq] = 1
            zeros_tensor[torch.arange(real_batch_size), target] = 1
            zeros_tensor = zeros_tensor[:,:-1]
            neg_tensor = 1 - zeros_tensor
            batch_neg = torch.multinomial(
                neg_tensor, num_negtive_items, replacement=True
            )
            target_neg = batch_neg.to(device)

            # llm_emb getting
            if llm_all_emb != None:
                llm_emb = torch.zeros(seq.size(0), seq.size(1), 4096, device=device)
                mask = seq < llm_all_emb.size(0)
                llm_emb[mask] = llm_all_emb[seq[mask]]
                llm_emb = llm_emb.to(device)  
            else:
                llm_emb = None

            # model forward          
            model_output = model.forward(seq, len_seq, llm_emb)

            # bce loss
            target = target.view(args.batch_size, 1)
            target_neg = target_neg.view(args.batch_size, 1)
            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)
            pos_labels = torch.ones((args.batch_size, 1))
            neg_labels = torch.zeros((args.batch_size, 1))
            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            scores = scores.to(device)
            labels = labels.to(device)
            loss = bce_loss(scores, labels)

            # dros loss
            if args.alpha != 0:
                pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
                pos_scores_dro = torch.squeeze(pos_scores_dro)
                pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
                pos_loss_dro = torch.squeeze(pos_loss_dro)
                inner_dro = (torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1)
                            - torch.exp((pos_scores_dro / args.beta)) 
                            + torch.exp((pos_loss_dro / args.beta)))
                loss_dro = torch.log(inner_dro + 1e-24)
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            else:
                loss_all = loss

            if args.lam != 0:
                candidate = all_candidate[sample] # [1024,k]
                candidate = candidate[:,:args.candidate_topk]   
                # weight_rank
                _lambda = 1
                _K = args.candidate_topk
                weight_static = torch.arange(1, _K + 1, dtype=torch.float32)
                weight_static = torch.exp(-weight_static / _lambda) # 1/exp(r)
                weight_static = weight_static.unsqueeze(0) # [1, k]
                weight_static = weight_static.repeat(args.batch_size, 1)
                weight_rank = weight_static / torch.sum(weight_static, dim=1).unsqueeze(1)
                weight_rank = weight_rank.to(device)               
                # weight_com
                cf_rank_top = (-model_output).argsort(dim=1)[:, :_K].to(device) # candidate [1024, k]
                common_tensor = torch.zeros_like(candidate).to(device)
                common_mask = candidate.unsqueeze(2) == cf_rank_top.unsqueeze(1)
                common_tensor = common_mask.any(dim=2).int() + 1e-8
                weight_com = common_tensor.to(device)
                # weight_confidence
                candidate_confidence = llm_confidence[sample] # [1024,k]
                candidate_confidence = candidate_confidence[:,:_K]
                weight_confidence = torch.exp(-candidate_confidence) + 1e-8
                weight_confidence = weight_confidence / torch.sum(weight_confidence, dim=1).unsqueeze(1)
                weight_confidence = weight_confidence.to(device)
                # weight_fin
                weight_fin = args.gamma_position*weight_rank + args.gamma_confidence*weight_confidence + args.gamma_consistency*weight_com
                weight = weight_fin / torch.sum(weight_fin, dim=1).unsqueeze(1) # [1024,k]
                # distillation loss
                loss_all_rd = 0
                num_candidate = candidate.size(1)
                for i_ in range(num_candidate):
                    target = candidate[:, i_:i_+1] # [1024,1]
                    # bce loss
                    pos_scores = torch.gather(model_output, 1, target) # [1024,1]
                    neg_scores = torch.gather(model_output, 1, target_neg) # [1024,1]
                    pos_labels = torch.ones((args.batch_size, 1)).to(device) # [1024,1]
                    neg_labels = torch.zeros((args.batch_size, 1)).to(device) # [1024,1]
                    loss_bce_rd = -(pos_labels*torch.log(torch.sigmoid(pos_scores)) + (1-neg_labels)*torch.log(torch.sigmoid(1-neg_scores)))
                    if args.alpha != 0:
                        # dro loss
                        pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
                        pos_scores_dro = torch.squeeze(pos_scores_dro) # [1024,1]
                        pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
                        pos_loss_dro = torch.squeeze(pos_loss_dro) # [1024,1]
                        A = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1)
                        B = torch.exp((pos_scores_dro / args.beta))
                        C = torch.exp((pos_loss_dro / args.beta))
                        inner_dro_rd =  A - B + C
                        loss_dro_rd = torch.log(inner_dro_rd + 1e-24) # [1024]
                        # all loss
                        loss_all_rd +=  (weight[:, i_:i_+1]*loss_bce_rd).mean() + args.alpha * (weight[:, i_:i_+1]*loss_dro_rd).mean() 
                    else:
                        loss_all_rd +=  (weight[:, i_:i_+1]*loss_bce_rd).mean()
                loss_all = loss_all + args.lam * (loss_all_rd)

            if torch.isnan(loss_all).any():
                print('loss is nan!!!')
                exit(-1)
            loss_all.backward()
            optimizer.step()
            
        if True:
            step = i + 1
            e_epoch = time.time()
            print(f"the loss in {step}th step is: {loss_all}, train this epoch cost {e_epoch-s_epoch} s")            
            if step % 1 == 0:
                print('VAL PHRASE:')
                val_path = os.path.join(data_directory, 'val_data.csv')
                prediction, hr_list, ndcg_list = myevaluate(model, val_path, device, llm_all_emb)
                print('TEST PHRASE:')
                test_path = os.path.join(data_directory, 'test_data.csv')
                s_test = time.time()
                prediction, hr_list, ndcg_list = myevaluate(model, test_path, device, llm_all_emb)
                e_test = time.time()
                if ndcg_list[-1] > best_ndcg20:
                    patient = 0 
                    best_ndcg20 = ndcg_list[-1]
                    best_hr20 = hr_list[-1]
                    best_hr_list_result = hr_list
                    best_ndcg_list_result = ndcg_list
                    best_step = step
                    best_prediction = prediction
                else:
                    patient += 1 
                print(f'patient={patient}, BEST STEP:{best_step}, BEST NDCG@20:{best_ndcg20}, BEST HR@20:{best_hr20}, test cost:{e_test-s_test}s')

                if patient >= 10:
                    e = time.time()
                    cost = (e - s)/60
                    print(f'=============early stop=============')
                    print(f'cost {cost} min')
                    exit(0)
