
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels,memory_features=None, memory_labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if memory_features==None and memory_labels == None:
          #print(labels.size())
          labels = labels.contiguous().view(-1, 1)
          #print(labels.size())
          anchor_feature = features
          mask = torch.eq(labels, labels.T).float().to(device)
          anchor_count = features.shape[0]
          contrast_count = anchor_count
          contrast_feature = anchor_feature
          logits_mask = torch.ones_like(mask).to(device)
          self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          logits_mask[:,:mask.size()[0]] = logits_mask[:,:mask.size()[0]].clone() * self_contrast_mask.to(device)
        elif memory_features!=None and  memory_labels!=None:
          anchor_count = features.shape[0]
          anchor_feature = features
          labels = labels.contiguous().view(-1, 1)
          memory_labels = memory_labels.contiguous().view(-1, 1)
          memory_count = memory_features.size()[0]
          contrast_count = anchor_count + memory_features.size()[0]
          contrast_labels = torch.cat([labels,memory_labels])
          mask = torch.eq(labels, contrast_labels.T).float().to(device)
          positive_mask = torch.eq(labels, labels.T).float().to(device)
          #filter_mask = torch.zeros((anchor_count, memory_count))
          #mask = torch.cat((positive_mask, filter_mask.to(device)), dim=1)
          memory_mask = 1 - torch.eq(labels, memory_labels.T).float().to(device)
          contrast_feature = torch.cat([anchor_feature, memory_features]).detach()
          #self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          #logits_mask = torch.cat((self_contrast_mask.to(device), memory_mask.to(device)),dim=1)
          logits_mask = torch.ones_like(mask).to(device)
          self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          logits_mask[:,:mask.size()[0]] = logits_mask[:,:mask.size()[0]].clone() * self_contrast_mask.to(device)
          #exit()
      

        # compute logits
        anchor_norm = torch.norm(anchor_feature,dim=1)
        contrast_norm = torch.norm(contrast_feature,dim=1)
        anchor_feature = anchor_feature/(anchor_norm.unsqueeze(1))
        contrast_feature = contrast_feature/(contrast_norm.unsqueeze(1))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # torch.matmul(anchor_norm, contrast_norm.T)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        #logits = anchor_dot_contrast
        # tile mask
        
        '''
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange( contrast_count * anchor_count).view(-1, 1).to(device),
            0
        )
        '''
        mask = mask * logits_mask
        nonzero_index = torch.where(mask.sum(1)!=0)[0]
        if len(nonzero_index) == 0:
          return torch.tensor([0]).float().to(device)
        # compute log_prob
        mask = mask[nonzero_index]
        logits_mask = logits_mask[nonzero_index]
        logits = logits[nonzero_index]
        exp_logits = torch.exp(logits) * logits_mask
        #exp_logits = logits * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #log_prob = logits/exp_logits.sum(1, keepdim=True)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / (self.base_temperature) ) * mean_log_prob_pos
        #loss =  (self.temperature / (self.base_temperature) ) * mean_log_prob_pos
        loss = loss.mean()
        return loss
