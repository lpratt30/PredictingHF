import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset

class ReadmissionCNN(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, embedding_dim):
        super().__init__()
        self.training = True

        # Since we already have embeddings, we won't use an embedding layer
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layers with in_channels=1 because our input is not an image but an embedded text
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1,
                                              out_channels=f,
                                              kernel_size=(fs, embedding_dim))
                                    for (fs,f) in zip(filter_sizes, n_filters)
                                    ])

        # Fully connected layer
        #self.fc = nn.Linear(sum(n_filters), output_dim)
        
        self.fc1 = nn.Linear(sum(n_filters), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)

        # # Dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, embedded):
        #self.activation_func = nn.LeakyReLU(0.05)
        self.activation_func = nn.ReLU()
        # embedded is already a tensor with size [batch size, notes len, emb dim]
        # We add an extra dimension for the "channel" which is 1 here
        #embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, notes len, emb dim]

        # Convolution and pooling layers
        conved = [self.activation_func(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # Max pooling over time
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        # Concatenate the pooled features from different filter sizes
        cat = self.dropout(torch.cat(pooled, dim=1))

        ## THERE IS NO NEED TO EXPLICITLY BYPASS DROPOUT FOR VALIDATION AS IT'S TAKEN CARE BY model.train() AND model.eval() FUNCTIONS AUTOMATICALLY
        #cat = None
        #if self.training:
        #    cat = self.dropout(torch.cat(pooled, dim=1))
        #else:
        #    cat = torch.cat(pooled, dim=1)

        ##cat = [batch size, n_filters * len(filter_sizes)]
        x = self.dropout(self.activation_func(self.fc1(cat))) 
        x = self.dropout(self.activation_func(self.fc2(x))) 
        # Fully connected layer
        return self.fc3(x)
    
        # Fully connected layer
        #return self.fc(cat)

class ReadmissionCNN_Enhanced(nn.Module):
    def __init__(self,n_filters, filter_sizes, output_dim, dropout, embedding_dim):
        super().__init__()
        
        # Since we already have embeddings, we won't use an embedding layer
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define convolutional layers independently
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters[0], kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=n_filters[1], kernel_size=(filter_sizes[1], embedding_dim))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=n_filters[2], kernel_size=(filter_sizes[2], embedding_dim))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=n_filters[3], kernel_size=(filter_sizes[3], embedding_dim))

        # Define dropout layers for each convolutional layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Define pooling layers for each convolutional layer
        # Assuming global max pooling, so no need to define them here

        # Fully connected layer
        #self.fc1 = nn.Linear(sum(n_filters), output_dim)
        
        self.fc1 = nn.Linear(sum(n_filters), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, embedded):
        self.activation_func = nn.LeakyReLU(0.05)
        
        #print(f'embed dimension before unsqueeze {embedded.shape}')
        conved1 = F.relu(self.conv1(embedded)).squeeze(3)
        pooled1 = F.max_pool1d(conved1, conved1.shape[2]).squeeze(2)
        dropped1 = self.dropout1(pooled1)

        # Apply second convolutional layer, ReLU, dropout, and pooling
        conved2 = F.relu(self.conv2(embedded)).squeeze(3)
        pooled2 = F.max_pool1d(conved2, conved2.shape[2]).squeeze(2)
        dropped2 = self.dropout1(pooled2)

        # Apply third convolutional layer, ReLU, dropout, and pooling
        conved3 = F.relu(self.conv3(embedded)).squeeze(3)
        pooled3 = F.max_pool1d(conved3, conved3.shape[2]).squeeze(2)
        dropped3 = self.dropout1(pooled3)

        # Apply third convolutional layer, ReLU, dropout, and pooling
        conved4 = F.relu(self.conv4(embedded)).squeeze(3)
        pooled4 = F.max_pool1d(conved4, conved4.shape[2]).squeeze(2)
        dropped4 = self.dropout4(pooled4)

        # Concatenate the pooled features from different convolutional layers
        cat = torch.cat((dropped1, dropped2, dropped3,dropped4), dim=1)
        # cat = [batch size, n_filters * len(filter_sizes)]
        
        x = self.dropout1(self.activation_func(self.fc1(cat))) 
        x = self.dropout2(self.activation_func(self.fc2(x))) 
        # Fully connected layer
        return self.fc3(x)

class VariedFilterReadmissionCNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, embedding_dim))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, embedding_dim))
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, embedded):
        #embedded = embedded.unsqueeze(1)

        x = F.relu(self.conv1(embedded))
        x = F.max_pool2d(x, (2, 1))
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 1))
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        #x = self.dropout(x)

        return self.fc(x)       

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
          
class TextDataset(Dataset):
    def __init__(self, tokenized_notes, labels):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame containing tokenized text and labels.
            embedding_model (Model): Pre-trained model to generate embeddings.
        """
        #print(type(tokenized_notes))
        #print(len(tokenized_notes))
        self.tokenized_notes = tokenized_notes
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.tokenized_notes)

       
    def __getitem__(self, idx):        
        #print(idx)
        #print(self.tokenized_notes.iloc[idx, 0])
        text = self.tokenized_notes.iloc[idx, 0]        
        label = self.labels[idx]
        # Retrieve the original DataFrame index
        original_index = self.tokenized_notes.iloc[idx,1]
        return text, label, original_index          
