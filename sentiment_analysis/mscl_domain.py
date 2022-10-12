import json
import torch
import codecs
import operator
import re
import time
import os
import math
from random import shuffle
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from torch.utils.data import DataLoader
from memory_supcontrast import SupConLoss
import random
import argparse
import pandas as pd
from transformers import XLMRobertaTokenizer,XLMRobertaConfig,XLMRobertaModel,AutoTokenizer, AutoModel, AutoModelWithLMHead,XLMRobertaForMaskedLM
from transformers import RobertaTokenizer,RobertaConfig,RobertaModel, get_linear_schedule_with_warmup
from transformers import BertTokenizer,BertConfig,BertModel
from typing import Union, Iterable


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print ('GPU will be used')
else:
    print ("CPU will be used")
 
def load_json(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def load_raw(filename, domain):
    """
    load file for evaluation or distillation
    """
    with open(filename, 'r',encoding='utf-8') as F:
        data = []
        for line in F:
            data.append([domain] + [0,line.strip()])
        print ("number of raw documents is", len(data))
        return data
def load_train(filename, domain):
    """
    load multiple files for training
    """
    data = []
    with open (filename, 'r',encoding='utf-8') as F:
        for line in F:
            data.append([domain]+line.split('\t'))
        print ("number of training pairs is", len(data))

    return data


def load_sst(filename):
    """
    load multiple files for training
    """
    data = []
    with open (filename, 'r',encoding='utf-8') as F:
        next(F)
        for line in F:
            data.append(line.split('\t'))
        print ("number of training pairs is", len(data))

    return data


def load_test(filename):
    """
    load file for evaluation or testing
    """
    with open(filename, 'r',encoding='utf-8') as F:
        data = []
        for line in F:
            data.append(line.split('\t'))
        print ("number of testing pairs is", len(data))
        return data


def xlm_r_sst(data,tokenizer,max_length):
    """
    obtain pretrained xlm_r representations
    """

    embedded_data=[]
    for pair in data:
        #tokens = tokenizer.tokenize(pair[1],max_length=max_length,pad_to_max_length=True)
        encoded = torch.LongTensor(tokenizer.encode(pair[0],max_length=max_length, truncation = True, pad_to_max_length=True)).to(device)
        #tensor_ones = torch.ones([max_length])
        #tokens = tokenizer.encode(pair[1])
        #if list(tokens.size())[0] >= max_length:
        #    tokens = tokens[:max_length]
        #tensor_ones[:len(tokens)] = tokens
        tem_label = torch.tensor([float(pair[1])]).long().to(device)
        #print(tem_label.shape)
        #print(token_tensor.shape)
        new_pair = torch.cat((tem_label, encoded), dim=0)
        embedded_data.append(new_pair.unsqueeze(0))
    embedded_data = torch.cat(embedded_data)
    return embedded_data


def xlm_r_t(data,tokenizer,max_length):
    """
    obtain pretrained xlm_r representations
    """

    embedded_data=[]
    for pair in data:
        #tokens = tokenizer.tokenize(pair[1],max_length=max_length,pad_to_max_length=True)
        encoded = torch.LongTensor(tokenizer.encode(pair[1],max_length=max_length, truncation = True, pad_to_max_length=True)).to(device)
        #tensor_ones = torch.ones([max_length])
        #tokens = tokenizer.encode(pair[1])
        #if list(tokens.size())[0] >= max_length:
        #    tokens = tokens[:max_length]
        #tensor_ones[:len(tokens)] = tokens
        tem_label = torch.tensor([float(pair[0])]).long().to(device)
        #print(tem_label.shape)
        #print(token_tensor.shape)
        new_pair = torch.cat((tem_label, encoded), dim=0)
        embedded_data.append(new_pair.unsqueeze(0))
    embedded_data = torch.cat(embedded_data) 
    return embedded_data 

def xlm_r_train(data,tokenizer,max_length, domain_label_map):
    """
    obtain pretrained xlm_r representations
    """

    embedded_data=[]
    for pair in data:
        #tokens = tokenizer.tokenize(pair[1],max_length=max_length,pad_to_max_length=True)
        encoded = torch.LongTensor(tokenizer.encode(pair[2],max_length=max_length, truncation = True, pad_to_max_length=True)).to(device)
        #tensor_ones = torch.ones([max_length])
        #tokens = tokenizer.encode(pair[1])
        #if list(tokens.size())[0] >= max_length:
        #    tokens = tokens[:max_length]
        #tensor_ones[:len(tokens)] = tokens
        domain_label = torch.tensor([float(domain_label_map[pair[0]])]).long().to(device)
        tem_label = torch.tensor([float(pair[1])]).long().to(device)
        #print(tem_label.shape)
        #print(token_tensor.shape)
        new_pair = torch.cat((domain_label, tem_label, encoded), dim=0)
        embedded_data.append(new_pair.unsqueeze(0))
    embedded_data = torch.cat(embedded_data)
    return embedded_data 





def train(train_data,model,fa_module, classifier,gradient_accumulate_step, epoch_num):
    """
    function for training the classifier
    """

    global global_step
    global start
    global results_table
    global class_centroids
    global momentum_embedding
    global memory_bank
    # Train the model
    train_loss = 0
    contrast_loss = 0
    intra_class_loss = 0
    train_acc = 0
    domain_acc = 0
    optimizer.zero_grad()
     
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    n_batches = len(data) 
    for i, pairs in enumerate(data):
        # print (pairs.size())
        text_a = pairs[ :, 2:params.max_length+2].to(device)
        index = pairs[ :, 1:2].to(device)
        domain_label = pairs[ :, :1].to(device)
        text_a=torch.squeeze(text_a)
        index = torch.squeeze(index)
        domain_label = torch.squeeze(domain_label)
        # optim.zero_grad()
        #optimizer.zero_grad()
        text_a,  index = text_a.to(device),  index.to(device)
        #last_hidden=model.extract_features(text)
        #print(index)
        last_hidden_a=model(text_a)[0]
        last_hidden_cls = last_hidden_a[:,0,:]#Classifier representation
        last_hidden_avg = torch.mean(last_hidden_a, dim=1)#Classifier representation
        last_hidden_cls = fa_module(last_hidden_cls)
        #last_hidden_avg = fa_module(last_hidden_avg)
      
        #print(index)
        #hinge_loss = 0
           #print(cls_centroids[cls_id]['centroid'].shape)
        #first_token_hidden = last_hidden[:,0,:]
        #contrast_input = torch.stack([last_hidden_cls, last_hidden_cls],dim=1)
        contrast_input = last_hidden_cls
        output = classifier(last_hidden_cls)

       
        #supcon_loss = contrast_criterion(last_hidden_cls,index)
        ce_loss = criterion(output, index)
        divisor = 24
       
        batch_prediction = output.argmax(1)
        inter_loss = 0 
        mmd_loss = 0 
        intra_loss = 0
        batch_cluster_loss = 0

        #moco_loss = MocoLoss(batch_mean_feature_list, class_centroids, moco_distance)     
        divisor = 24
        #if torch.unique(index, return_counts=True)[1].min()<=2:
        #  supcon_loss = 0
        if memory_bank == None:
          #supcon_loss = 0
          supcon_loss = contrast_criterion(contrast_input,index)
        elif memory_bank != None:
          #supcon_loss = moco_loss(batch_example, memory_bank, params.nclass)
          memory_label = memory_bank[:,:1].squeeze()
          memory_feature = memory_bank[:, 1:]
          supcon_loss = contrast_criterion(contrast_input,  index, memory_feature, memory_label)
     
        batch_example = torch.cat([index.unsqueeze(1), last_hidden_cls], dim = 1 )
        enqueue_and_dequeue(batch_example, params.memory_bank_size) 
        #contrast_loss = moco_loss(batch_example, memory_bank, params.nclass)  
        #if torch.unique(index, return_counts=True)[1].min() <=2:
        #  contrast_loss = 0
        #else:
        #  contrast_loss = contrast_criterion(contrast_input, index)
        
        composite_loss =  params.lambda_ce  * ce_loss + params.lambda_moco * supcon_loss 
        #composite_loss =  params.lambda_ce  * ce_loss
        train_loss  +=  ce_loss.item()  
        contrast_loss += supcon_loss 
        if global_step%100 == 0:
        
           #print(supcon_loss.requires_grad)
           print(f'CE loss: {params.lambda_ce  * ce_loss: .4f}')
           print(f'Contrast loss: {supcon_loss: .4f}')
        #composite_loss.backward(retain_graph=True)
        composite_loss.backward()
        if (i+1)%gradient_accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(fa_module.parameters(), params.grad_clip_norm)
            #torch.nn.utils.clip_grad_norm_(projection.parameters(), params.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()
        # optim.step()
        train_acc += (output.argmax(1) == index).sum().item()
    return train_loss / n_batches, contrast_loss / n_batches ,train_acc / (1 * len(train_data))


def test(test_data, model, fa_module, classifier):
    """
    function for evaluating the classifier
    """

    loss = 0
    acc = 0
    data = DataLoader(test_data, batch_size=BATCH_SIZE)
    pred = []
    pred_probs = []
    ground = []
    n_batches = len(data)
    model.eval()
    #fa_module.eval()
    classifier.eval()
    for i, pairs in enumerate(data):
        text = pairs[ :, 1:].to(device)
        index = pairs[ :, :1].to(device)
        text=torch.squeeze(text)
        index = torch.squeeze(index)
        index = index.long()
        
        text, index = text.to(device), index.to(device)
        with torch.no_grad():

            last_hidden=model(text)[0]
            last_hidden_cls = last_hidden[:,0,:]
            last_hidden_cls = fa_module(last_hidden_cls)
            output=classifier(last_hidden_cls)
            output=softmax(output)

            l = criterion(output, index)
            loss += l.item()
            acc += (output.argmax(1) == index).sum().item()
            pred_probs.append(output.detach().cpu().numpy())
            pred.append(output.argmax(1).detach().cpu().numpy())           
            ground.append(index.detach().cpu().numpy())
    return loss / n_batches, acc / len(test_data)


def cosine_sim(a,b):
    return np.dot(a,b) / ( (np.dot(a,a) **.5) * (np.dot(b,b) ** .5) )

class FAM(nn.Module):
    def __init__(self, embed_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(embed_size, hidden_size)
        #self.init_weights()
    def init_weights(self):
        initrange = 0.2
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text):#, return_att = False):
        batch,  dim = text.size()
        feat = self.fc(torch.tanh(self.dropout(text.view(batch, dim))))
        feat = F.normalize(feat, dim=1)
        return feat

class SupConHead(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', dim_in=1024, feat_dim=256):
        super(SupConHead, self).__init__()
        #model_fun, dim_in = model_dict[name]
        #self.encoder = model_fun()
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                #nn(inplace=True),
                nn.Tanh(),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        #feat = self.encoder(x)
        feat = F.normalize(self.head(x), dim=1)
        return feat

class Projection(nn.Module):
    def __init__(self, hidden_size, projection_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, projection_size)
        self.ln = nn.LayerNorm(projection_size)
        self.bn = nn.BatchNorm1d(projection_size)
        self.init_weights()
    def init_weights(self):
        initrange = 0.01
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text):#, return_att = False):
        #text = text.view()
        batch,  dim = text.size()

        return self.ln(self.fc(torch.tanh(text.view(batch, dim))))


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class CosineDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1) 

    def forward(self, feature, centroid):
        cos_distance = 1 - self.cos(feature, centroid)
        #cos_distance =  self.cos(feature, centroid)
        #print(cos_distance.size())
        cos_distance = torch.mean(cos_distance, dim=0)    
        return cos_distance

class L2Distance(nn.Module):
    def __init__(self):
        super().__init__()
        #self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid):
        L2_distance = (feature - centroid)**2
        #cos_distance =  self.cos(feature, centroid)
        #print(cos_distance.size())
        L2_distance = torch.sum(L2_distance, dim=1)
        L2_distance = torch.mean(L2_distance, dim=0)
        return L2_distance



class InterCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid):
        cos_distance = 1 + self.cos(feature, centroid)
        #cos_distance =  self.cos(feature, centroid)
        #print(cos_distance.size())
        cos_distance = torch.mean(cos_distance, dim=0)
        return cos_distance

class PositiveContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid, temp=1.0):
        feature_norm = torch.norm(feature, dim=1)
        centroid_norm = torch.norm(centroid, dim=1)
        batch_dot_product = torch.bmm(feature.view(feature.size()[0], 1, feature.size()[1]), centroid.view(centroid.size()[0], centroid.size()[1], 1))
        batch_dot_product =  batch_dot_product.squeeze()/(feature_norm * centroid_norm)
        batch_dot_product = torch.mean(batch_dot_product)     
        
        return  1 - 1*batch_dot_product

class MocoLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    def forward(self, batch_examples, memory_bank, nclass):
        loss = 0
        for class_id in range(nclass):
          feature_index = torch.where(batch_examples[:,:1] == class_id)
          
          feature = batch_examples[feature_index[0]][:,1:]
          if feature.size()[0] <= 1 :
            continue
          feature_norm = torch.norm(feature, dim=1)
          feature = feature / feature_norm.unsqueeze(1)
          mask = torch.diag(torch.ones((feature.size()[0])))
          #print(mask)
          mask = (1 - mask).to(device)
          #print(mask)
          feature_dot_feature = torch.div(torch.matmul(feature, feature.T), self.temperature)
          if torch.isnan(feature_dot_feature).any():
            print("feature:")
            print(feature)
            exit()
          #logits_max, _ = torch.max(feature_dot_feature, dim=1)
          #feature_dot_feature = feature_dot_feature * mask.to(device)
          #feature_dot_feature = feature_dot_feature
          negative_index = torch.where(memory_bank[:,:1] != class_id)
          #print(negative_index)
          #exit() 
          negative_examples = memory_bank[negative_index[0]][:,1:].detach()
          negative_norm = torch.norm(negative_examples, dim=1)
          negative_examples = negative_examples / negative_norm.unsqueeze(1)
          
          feature_dot_negative = torch.div(torch.matmul(feature, negative_examples.T), self.temperature)
          logits_mask_positive = torch.ones(feature_dot_negative.size()).to(device)
          logits_mask_negative = torch.zeros(feature_dot_negative.size()).to(device)
          logits = torch.cat([feature_dot_feature,feature_dot_negative],dim=1)
          positive_mask = torch.cat([mask, logits_mask_negative], dim=1)
          logits_mask = torch.cat([mask, logits_mask_positive], dim=1)
          logits = logits * logits_mask
          #print(logits_mask)
          #print(positive_mask)
          logits_max, _ = torch.max(logits, dim=1, keepdim=True)
          logits = (logits - logits_max.detach()).to(device)
          #logits = logits_copy
          
          exp_logits = torch.exp(logits)
          
          c_loss = -1 * (logits*positive_mask).sum(1)/(logits*logits_mask).sum(1)
          #c_loss = -1 * (exp_logits*positive_mask).sum(1)/(exp_logits*logits_mask).sum(1)
          
          sums = exp_logits.sum(1,keepdims=True)
          log_prob = logits - torch.log(sums ) 
         
          mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
          #loss += c_loss.mean()
          loss += -1 * (0.07) * mean_log_prob_pos.mean()
          #print(log_pro)
          #print(positive_mask)
          #print(loss)
          
        return  loss


class NegativeContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid, temp=1.0):
        feature_norm = torch.norm(feature, dim=1)
        centroid_norm = torch.norm(centroid, dim=1)
        batch_dot_product = torch.bmm(feature.view(feature.size()[0], 1, feature.size()[1]), centroid.view(centroid.size()[0], centroid.size()[1], 1))
        batch_dot_product = batch_dot_product.squeeze()/(feature_norm * centroid_norm)
        batch_dot_product = torch.mean(batch_dot_product)     
        #dot_product = torch.div( torch.matmul(feature, centroid.T), torch.matmul(feature_norm, centroid_norm.T))
        #dot_product = torch.div( torch.matmul(feature, centroid.T), temp)
        #batch_dot_product = torch.exp(batch_dot_product)
        #batch_dot_product = torch.log(batch_dot_product)
        #batch_dot_product = torch.mean(batch_dot_product.size())
        
        return batch_dot_product


class PairwiseCosineDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)
 
    def forward(self, feature):
        mean_distance=[]
        for cls in range(params.nclass):
            access = params.nclass * [True]
            access[cls] = False
            cls_cos_dist = 1 + self.cos(torch.stack((params.nclass - 1)* [feature[cls]],dim=0), feature[access])
            mean_distance.append(torch.mean(cls_cos_dist, dim=0)/2)
        mean_distance = torch.stack(mean_distance) 
        
        mean_distance = torch.mean(mean_distance, dim=0)
        return mean_distance

class PairwiseContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature):
        nclass = feature.size()[0]
        feature_norm = torch.norm(feature, dim = 1)
        inter_class_mask = 1-torch.diag(torch.ones((nclass)))
        norm_dot_norm = torch.matmul(feature_norm,feature_norm.T)
        class_dot_class = torch.matmul(feature,feature.T)
        class_dot_class = torch.div(class_dot_class, norm_dot_norm)
        class_dot_class = inter_class_mask.to(device) * class_dot_class
        mean_distance = torch.sum(class_dot_class) / (nclass**2 - nclass)
        mean_distance = torch.log(torch.exp(mean_distance))
        return mean_distance




class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid, dummy):
        cos_similarity = self.cos(feature, centroid)
        #print(cos_distance.size())
        #exit()
        cos_similarity = torch.mean(cos_similarity, dim=0)
        return cos_similarity



def BatchMeanFeature(feature, index, nclass):
    batch_mean_feature_list = []
    batch_mean_feature_list_detach = []
    #batch_mean_feature_tensor = torch.zeros((nclass, feature.size()[1]), requires_grad=True)
    batch_mean_feature_tensor = []
    for cls_id in range(nclass):
       cls_id_idx = torch.where(index == cls_id)
       if len(cls_id_idx) > 0:
           batch_centroid = feature[cls_id_idx].detach()
           batch_mean_feature_list.append(feature[cls_id_idx])
           batch_mean_feature_list_detach.append(batch_centroid)
           batch_mean_feature_tensor.append(torch.mean(feature[cls_id_idx], dim=0))   ##mean feature of class cls_id within one batch
       else:
           batch_mean_feature_list.append(None)
    batch_mean_feature_tensor=torch.stack(batch_mean_feature_tensor, dim=0)
    return batch_mean_feature_tensor, batch_mean_feature_list, batch_mean_feature_list_detach



def InitMeanFeature(momentum_embedding, nclass):
    global class_centroids
    temp_tensor = []
    for cls_id in range(nclass):
        class_centroids[cls_id] = [x for x in class_centroids[cls_id] if x != None]
        print(torch.cat(class_centroids[cls_id], dim=0).size())
        init_mean_feat = torch.mean(torch.cat(class_centroids[cls_id], dim=0),dim=0)
        temp_tensor.append(init_mean_feat)
        class_centroids[cls_id] = []
    temp_tensor = torch.stack(temp_tensor, dim=0)
    momentum_embedding.requires_grad = False
    momentum_embedding.weight.data = temp_tensor
    return momentum_embedding

def IntraClassLoss(batch_mean_feature_list, momentum_embedding, distance_metric):
    nclass = len(batch_mean_feature_list)
    b_intra = 0
    for cls_id in range(nclass):
        if  batch_mean_feature_list[cls_id].size()[0] > 0:
            num_samples = batch_mean_feature_list[cls_id].size()[0]
            #print(num_samples)
            #print(momentum_embedding.weight.data[cls_id].size())
            b_intra += distance_metric(batch_mean_feature_list[cls_id],  torch.stack(num_samples * [momentum_embedding.weight.data[cls_id]], dim = 0) )/nclass
    return b_intra




def InterClassLoss(batch_mean_feature_list, momentum_embedding, inter_distance_metric):
    nclass = len(batch_mean_feature_list)
    b_inter = 0
    for cls_id in range(nclass):
        if batch_mean_feature_list[cls_id].size()[0]>0:
            access = torch.ones((nclass))
            access[cls_id] = 0
            inter_class = torch.where(access == 1)[0].numpy().tolist()
            num_samples = batch_mean_feature_list[cls_id].size()[0]
            for i_cls in  inter_class:
                b_inter += inter_distance_metric(batch_mean_feature_list[cls_id],  torch.stack(num_samples * [momentum_embedding.weight.data[i_cls]], dim = 0) )/(nclass*len(inter_class))
    return b_inter




class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=257, K=1000, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.register_buffer("queue", torch.randn(dim, K))


def enqueue_and_dequeue(batch_examples, memory_bank_size):
    global memory_bank
    if memory_bank==None:
      memory_bank = batch_examples
    else:
      memory_bank = torch.cat([memory_bank, batch_examples.detach()] , dim=0)
      if memory_bank.size()[0] > memory_bank_size:
        memory_bank = memory_bank[-memory_bank_size:,:]
        
    

def UpdateMomentum(momentum_embedding, ema_updater, nclass):
    global class_centroids
    temp_tensor = []
    for cls_id in range(nclass):
        class_centroids[cls_id] = [x for x in class_centroids[cls_id] if x != None]
        init_mean_feat = torch.mean(torch.cat(class_centroids[cls_id], dim=0),dim=0)
        temp_tensor.append(init_mean_feat)
        class_centroids[cls_id] = []
    temp_tensor = torch.stack(temp_tensor, dim=0)
    update_embedding.requires_grad = False
    update_embedding.weight.data = temp_tensor
    update_moving_average(ema_updater, momentum_embedding, update_embedding)        
    return momentum_embedding

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        






class Classifier(nn.Module):
    def __init__(self, hidden_size, num_class, hidden_dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, feature):
        #print (feature.size())
        #print (feature)

        return self.fc(torch.tanh(feature))


def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        help="Experiment dump path")
    parser.add_argument("--nclass", type = int, default = 2)
    parser.add_argument("--nepochs", type = int, default = 10)
    parser.add_argument("--train_limited", type = bool, default = False)
    parser.add_argument("--save_model", type = bool, default = False)
    parser.add_argument("--batch_size", type = int, default = 12)
    parser.add_argument("--gradient_acc_step", type = int, default = 1)
    parser.add_argument("--projection_size", type = int, default = 512)
    parser.add_argument("--hidden_size", type = int, default = 256)
    parser.add_argument("--embedding_size", type = int, default = 1024)
    parser.add_argument("--max_length", type = int, default = 180)
    parser.add_argument("--warmup_step", type = int, default = 600)
    parser.add_argument("--skip_step", type = int, default = 300)
    parser.add_argument("--eval_step", type = int, default = 100)
    parser.add_argument("--m_update_interval", type = int, default = 10)
    parser.add_argument("--topk", type = int, default = 10)
    parser.add_argument("--seed", type = int, default = 24)
    parser.add_argument("--topk_use", type = int, default = 10)
    parser.add_argument("--valid_size", type = int, default = 200)
    parser.add_argument("--memory_bank_size", type = int, default = 200)
    parser.add_argument("--train_num", type = int, default = 10000)
    parser.add_argument("--train_few_shot", type = int, default = 0)
    parser.add_argument("--temp", type = float, default = 0.07)
    parser.add_argument("--hidden_dropout_prob", type = float, default = 0.0)
    parser.add_argument("--lambda_intra", type = float, default = 0.0)
    parser.add_argument("--lambda_inter", type = float, default = 0.0)
    parser.add_argument("--centroid_inter_loss", type = float, default = 0.0)
    parser.add_argument("--lambda_ce", type = float, default = 1.0)
    parser.add_argument("--lambda_kl", type = float, default = 0.0)
    parser.add_argument("--lambda_nce", type = float, default = 0.0)
    parser.add_argument("--lambda_supcon", type = float, default = 0.0)
    parser.add_argument("--lambda_adv", type = float, default = 0.0)
    parser.add_argument("--lambda_mmd", type = float, default = 0.0)
    parser.add_argument("--lambda_moco", type = float, default = 0.0)
    parser.add_argument("--centroid_decay", type = float, default = 0.99)
    parser.add_argument("--weight_decay", type = float, default = 0.98)
    parser.add_argument("--grad_clip_norm", type = float, default = 1.0)
    parser.add_argument("--lr", type = float, default = 5e-6)
    parser.add_argument("--model_name", type = str)
    parser.add_argument("--source_domain", type = str)
    parser.add_argument("--language_model", type = str,default='xlmr',help="Pre-trained language model: xlmr|roberta")
    parser.add_argument("--test_path", type = str)
    parser.add_argument("--valid_path", type = str)
    #parser.add_argument("--save_mi_path", type = str)
    parser.add_argument("--dataset", type = str,default='mtl-dataset')
    parser.add_argument("--log", type = str,default='multi-domain-log')

    return parser


if __name__ == '__main__':
    
    parser = get_parser()
    params = parser.parse_args()
    print(params)
    set_seed(params.seed)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    #model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    model = AutoModel.from_pretrained(params.model_name).to(device)
    #model = torch.hub.load('pytorch/fairseq', 'xlmr.large').to(device)
    model.train()
    print(type(model))
    test_domains = ['book', 'dvd', 'kitchen', 'electronics']
    max_length = params.max_length
    data_dir = 'data/benchmark/amazon/'

    domain_label_map = {}
    i = 0
    source_domains = params.source_domain.split()
    for domain in source_domains:
      domain_label_map[domain] = i
      i += 1
    print(domain_label_map)
    train_data = {}
    valid_data = {}
    test_data = {}
    raw_data = {}
    total_valid = []
    total_train = []
    toral_raw = []
    train_domains = params.source_domain.split()
    for domain in train_domains:
      train_data[domain] = load_train(data_dir + '{}/train'.format(domain), domain)
      if params.train_limited:
        random.shuffle(train_data[domain])        
        train_data[domain] = train_data[domain][:int(len(train_data[domain])/len(train_domains))]
      if params.train_few_shot != 0 :
        random.shuffle(train_data[domain])        
        train_data[domain] = train_data[domain][:params.train_few_shot]
  
      train_data[domain] = xlm_r_train(train_data[domain],tokenizer,max_length, domain_label_map)
      valid_data[domain] = load_test(data_dir + '{}/test'.format(domain))
      valid_data[domain] = xlm_r_t(valid_data[domain],tokenizer,max_length)
      raw_data[domain] = load_raw(data_dir + '{}/raw'.format(domain), domain)
      raw_data[domain] = xlm_r_train(raw_data[domain],tokenizer,max_length, domain_label_map)
      total_train.append(train_data[domain])
      total_valid.append(valid_data[domain])
      test_domains.remove(domain)

    total_train = torch.cat(total_train, dim=0)
    print(f'Total Training Examples: {total_train.size()[0]:d}')
    total_valid = torch.cat(total_valid, dim=0)
    target_test = {}
    for test_domain in test_domains:
      test_data[test_domain] = load_test(data_dir + '{}/test'.format(test_domain))
      test_data[test_domain] = xlm_r_t(test_data[test_domain],tokenizer,max_length)    

    
    
    ema_updater = EMA(params.centroid_decay)
    BATCH_SIZE=params.batch_size
    gradient_accumulate_step = params.gradient_acc_step
    FA_module = FAM(params.embedding_size, params.hidden_size, params.hidden_dropout_prob).to(device)
    #FA_module = SupConHead(feat_dim=params.hidden_size).to(device)
    projection = Projection(params.hidden_size, params.projection_size).to(device)
    classifier = Classifier(params.hidden_size, params.nclass, params.hidden_dropout_prob).to(device)
    domain_classifier = Classifier(params.hidden_size, len(source_domains), params.hidden_dropout_prob).to(device)
    softmax = nn.Softmax(dim=1)
    l1_criterion = torch.nn.SmoothL1Loss(reduction='mean').to(device)
    hinge_criterion = nn.HingeEmbeddingLoss(reduction = 'mean').to(device)
    #distance_metric = nn.CosineSimilarity(dim = 1)
    #distance_metric = L2Distance()
    distance_metric = PositiveContrastLoss()
    moco_loss = MocoLoss(params.temp)
    #distance_metric = nn.MSELoss()
    #pairwise_dist = PairwiseCosineDistance()
    pairwise_dist = PairwiseContrastLoss()
    inter_distance_metric = NegativeContrastLoss()
    cos_embedding_loss = nn.CosineEmbeddingLoss(margin = 0.0, reduction='mean').to(device)
    #distance_metric = nn.MSELoss(reduction='mean').to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    domain_criterion = torch.nn.CrossEntropyLoss().to(device)
    

    momentum_embedding = nn.Embedding(params.nclass, params.hidden_size)
    update_embedding = nn.Embedding(params.nclass, params.hidden_size)

    contrast_criterion = SupConLoss(temperature=params.temp).to(device)
    #self_contrast_criterion = SupConLoss(contrast_mode='all', temperature=params.temp).to(device)
    optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters()  if p.requires_grad == True], 'weight_decay': 0.0 } ,{'params': domain_classifier.parameters()},{'params': FA_module.parameters()}, {'params': classifier.parameters()}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=params.lr)
    #optimizer = torch.optim.Adam([ {'params': classifier.parameters()},  {'params':model.parameters()}], lr=params.lr)
    t_total = len(DataLoader(total_train, batch_size=BATCH_SIZE, shuffle=True))* params.nepochs / params.gradient_acc_step 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=params.warmup_step, num_training_steps=t_total)
    index_map_3_class = {0:0, 1:0, 2:1, 3:2, 4:2}   
    memory_bank = None
    #memory_bank = torch.randn((6,129)).to(device)
    results_table = pd.DataFrame(columns=['test_domain','epoch','valid_loss','valid_acc','test_loss','test_acc'])
    start = time.time()

    #global_step
    global_step = 0 
    for i in range(params.nepochs):
      train_loss, contrast_loss, train_acc = train(total_train, model,FA_module, classifier,gradient_accumulate_step, i)
      valid_loss, valid_acc = test(total_valid,model, FA_module, classifier)
  

      end = time.time()

    
      print(f'\tEpoch: {i+1:.4f}\t|\tTime Elapsed: {end-start:.1f}s')
      print(f'\tCE: {train_loss:.4f}(train) \t| Contrast_loss: {contrast_loss:.4f} \tAcc: {train_acc * 100:.2f}%(train) \t|\t')
      print(f'\tLoss: {valid_loss:.4f}(valid_loss)\t|\tAcc: {valid_acc * 100:.2f}%(valid_acc)')
      for test_domain in test_domains: 
        test_loss, test_acc = test(test_data[test_domain],model, FA_module, classifier)
        print(f'\tDomain: {test_domain:s} | Loss: {test_loss:.4f}(test_loss)\t|\tAcc: {test_acc * 100:.2f}%(test_acc)')
        if (i+1) >= 5:
          results_table = results_table.append({'test_domain': test_domain,'epoch':i+1,'valid_loss':valid_loss,
              'valid_acc':valid_acc,'test_loss':test_loss,'test_acc':test_acc}, ignore_index=True)
      start = time.time()
    for domain_test in test_domains:
      domain_result = results_table[results_table['test_domain']==domain_test].copy()
      print(domain_result.sort_values(by=['valid_acc'], ascending=False).head())
    if params.save_model:
      torch.save({'model': model, 'fam': FA_module, 'classifier': classifier}, 'model_{}.pt'.format(domain_test))
