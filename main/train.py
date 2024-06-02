#Code taken from for training: https://www.analyticsvidhya.com/blog/2023/06/step-by-step-bert-implementation-guide/#h-implementation-of-bert
#Adapted for the purpose of this project and then additional code added for fine-tuning and redistribution of data 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Imports, requiremnets and specifications                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # Import the necessary modules:
import numpy as np #allows for mathematical functions on large sets of data
import pandas as pd #data analysis and data preparation for ML training
import torch #main PyTorch module that provides Neural-Network related functions such as layering, functions and optimizers (for training neural networks)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split #scikit-learn provides ML tools for efficient data analysis and data mining
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight #compute class weights with multiple params
import transformers #fast tokenizeer for transformer-based models like BERT
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW #optimizer from Hugging Face, AdamW includes weight decay regularization
import matplotlib.pyplot as plt #used for histogram
import os #for file and directory functions
import time
from model import BERT_Arch

#Modules needed for undersampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



# specify the GPU being used and if it's available, otherwise fall back on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') #will be used later 

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DATASET
#Access and splitting
# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load training dataset
df = pd.read_csv(os.path.join(script_dir, "dataset/train.csv"), usecols=['label', 'tweet'])


# check class distribution
df['label'].value_counts(normalize = True)


# split train dataset into train, validation and test sets, only keep the tweet column and label column 
train_text, temp_text, train_labels, temp_labels = train_test_split(df['tweet'], df['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=df['label'])


#split the split test data into validation and testing using the same columns 
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)

#----------------------------------------------------------------------------------------------------------
#UNDERSAMPLING
#There are a lot more tweets with the label 0 than there are with label 1
#This is causing reduced precision for label 1 and mislabelling most data as 0 even if it is hateful
#Tried oversampling but the precision decreased further (from 28% to 18%), may be due to the generation of synthetic data

# Combine training text and labels for undersampling
train_data = pd.concat([train_text, train_labels], axis=1)

# Undersample the majority class (label 0)
undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
train_data_resampled, _ = undersample.fit_resample(train_data.drop('label', axis=1), train_data['label'])

# Separate undersampled data into text and labels
train_text_resampled = train_data_resampled['tweet']
train_labels_resampled = train_data_resampled.index  # Assuming the index represents the label after undersampling

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
train_text_tfidf = tfidf_vectorizer.fit_transform(train_text_resampled)

# Train a model (Random Forest as an example)
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(train_text_resampled, train_labels_resampled)

# Evaluate the model on the validation set
val_preds = model.predict(val_text)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#HISTOGRAM (generating a histogram for the frequency of specific words used)

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins = 30)


# Save the histogram 
plt.hist(seq_len, bins=30, edgecolor='black')
plt.title('Histogram of Message Lengths in Training Set')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

# Save the plot as a PNG file
plt.savefig(os.path.join(script_dir, "histogram.png"))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#TENSOR FORMATION AND DATA PREPARATION FOR DEEP LEARNING

#TOKENIZING
# NLPs need data to be formatted to be understood by ML models. 
#So the texts (tweets) are tokenized and encoded
# Each set of data (training, validation and testing) is tokenized and validated
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(), #make a list
    max_length = 25, #assign max length
    padding=True, #makes sure all lengths are same
    truncation=True #truncate to make sure desired length is achieved 
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    padding=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    padding=True,
    truncation=True
)

#CONVERTING TO TENSORS
# convert tokenized lists to tensors (data structures used in deep learning) 
#"input_id" is generated when tokenizing and encoding 
#attention_mask  is used to indicate what to focus on during processing
#The non tokenized and encoded labels are also converted to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

#DATA LOADER
#Imported "TensorDataset, DataLoader, RandomSampler, SequentialSampler"

#define a batch size
batch_size = 32

# wrap tensors  into one dataset object
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training randomly, ensures diverse data
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training sequentially
#for validation, the data no longer needs to be diverse
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

#CHANGE PRE-TRAINED MODEL PARAMS FOR TASK 
# freeze all the parameters, adapts pretrained BERT model for task 
for param in bert.parameters():
    param.requires_grad = False    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#GPU ACCELERATION

#info: weight regularization is used during training neural networks to prevent overfitting. 
#weights are parameters of a model learned during training process. 
#weights are associated with the conncetion between neurons. 
#weight regularization basically means the prevention of long list of parameters
#weight regularization aids in the generalization of parameters to allow for new, unseen data

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

#"from transformers import AdamW"
# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-5)

#"from sklearn.utils.class_weight import compute_class_weight"
#compute the class weights
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(train_labels), y= train_labels)

print("Class Weights:", class_weights)

#WEIGHTS TO TENSOR CONVERSION
# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

# number of training epochs
epochs = 11

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FINE-TUNE 
#Training function that iterates over batches of data with forward and backward passes
##updates model parameters and computes training loss. 
#stores model predictions and returns average loss and predictions

#TRAINING
# function to train the model
def train():
    
    model.train() #set model to training mode
    #initiliaze total_loss and total_accuracy with a pre-assigned value of 0
    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        
        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        #unpacks batch into individual components
        sent_id, mask, labels = batch
        
        # clear previously calculated gradients 
        model.zero_grad()  

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on CPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
  
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

#----------------------------------------------------------------------------------------------------------------------
#initially the format_time() function was throwing errors "elapsed = format_time(time.time() - time.time())" (line 346)
#so created a function that essentially has the same output 

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    
    # Format as hh:mm:ss
    return str(time.strftime("%H:%M:%S", time.gmtime(elapsed_rounded)))


#-----------------------------------------------------------------------------------------
#EVALUATE MODEL ON VALIDATION DATA
# function for evaluating the model
# computes validation loss, stores model predictions and returns average loss and predictions
# dropout layers are deactivated ("model.eval()")

#VALIDATION
def evaluate():
    
    print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0
    
    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - time.time())
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        #unpack into individual components
        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            # accumulate the total_loss
            total_loss = total_loss + loss.item()

            # detach and move predictions to CPU, convert to NumPy
            #GPU no longer required + NumPy operations are better done on CPU
            preds = preds.detach().cpu().numpy()

            # append the predictions to the total_preds list
            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

#-----------------------------------------------------------------------------------------
#TRAINING AND VALIDATION NN MODELS FOR SPECIFIC NUMBER OF EPOCHS
#essential for training ML model iteratively over multiple passes
#allows NN to learn from the data due to repeated exposure 
#can monitor losses and increase precision
# set initial loss to infinite
best_valid_loss = float('inf')

#defining epochs
#info: epochs are a complete pass of training dataset through the algorithm. 
epochs = 11 #better use 11 epochs for better results (https://datascientest.com/en/epoch-an-essential-notion#:~:text=Generally%2C%20a%20number%20of%2011,to%20optimally%20modify%20the%20weights.)

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
#so for every training pass
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train() #train  function called to train model and compute training loss
    
    #evaluate model
    valid_loss, _ = evaluate() #evaluation function called to evaluate model on validation set and compute validation loss
    
    #save the best model
    #if current validation loss is lower (basically more precise model) then store the model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


#----------------------------------------------------------------------------------------------------------

    # Load weights of the best model
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))


#----------------------------------------------------------------------------------------------------------
#TESTING
# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()


# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

# ... (previous code)

from sklearn.feature_extraction.text import TfidfVectorizer

