import torch.nn as nn
from transformers import AutoModel
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NEURAL NETWORK (DEEP LEARNING)
#The BERT_Arch class has a forward function that takes tweet IDs and masks as inputs
class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert 
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)
      
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model, unecessary words are ignored
        #cls_hs represents the essence of an entire sentence 
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        #linear transformation      
        #passed through a dense layer that helps model understand important patterns
        #assign result to variable "x" 
        x = self.fc1(cls_hs)

        #the result of self.fc1() may have negative values since patterns are being analysed
        #self.relu turns those negative values into positive through mathematical calculations
        x = self.relu(x)

        #a few elements within tensor x will be turned to 0 randomly
        #a regularization technique to help prevent overfitting by introducing random noise in the training process
        #dropout only applied to hidden layer 1 (fc1())
        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
      
        # apply softmax activation
        x = self.softmax(x)

        return x