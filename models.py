import torch
import torch.nn as nn 
import torchvision.models
import torchtext
vgg = torchvision.models.vgg
glove = torchtext.vocab.GloVe(name="6B", dim=50)
mean = glove.vectors.mean(dim=0)
glove_vector_with_unk = torch.cat((glove.vectors,mean.unsqueeze(0)))

class caption_transformer(nn.Module):
    def __init__(self,num_heads):
        super(caption_transformer,self).__init__()
        self.cnn_emb = vgg.features
        self.cnn_layer = nn.Conv2d(512,128,1)
        self.word_emb = nn.Embedding.from_pretrained(glove_vector_with_unk)
        self.fc_layer = nn.Linear(50,128)      
        self.transformer_layer = nn.Transformer(128,num_heads,batch_first=True)  
        self.fc_layer2 = nn.Linear(128,50)
        self.positional_embedding = PositionalEncoding(0.1,128)
        
    def forward(self, inp, target):
        # embed the image
        emb_inp = torch.relu(self.cnn_layer(inp))
        # positional embedding
        emb_inp = torch.relu(self.positional_embedding(emb_inp))
        # embed the text
        emb_target = torch.relu(self.fc_layer(target))
        N = emb_inp.shape[0]
        dim = emb_inp.shape[1] 
        emb_inp = emb_inp.view(N,dim,-1)
        emb_inp = torch.transpose(emb_inp, 1,2)
        out = self.transformer_layer(emb_inp,emb_target)
        out = torch.relu(self.fc_layer2(out))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self,dropout_rate, input_dimensions):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        positional_embedding = torch.zeros(input_dimensions,7)
        even = torch.arange(0,input_dimensions,2)
        odd = torch.arange(1,input_dimensions,2)
        position = torch.arange(7)
        denominator = torch.float_power(10000,even/input_dimensions)
        positional_embedding[0::2] = torch.sin(position.unsqueeze(0)/denominator.unsqueeze(1))
        positional_embedding[1::2] = torch.cos(position.unsqueeze(0)/denominator.unsqueeze(1))
        horizontal_positional_embedding = positional_embedding
        vertical_positional_embedding = positional_embedding
        self.positional_embedding = horizontal_positional_embedding.unsqueeze(1) + vertical_positional_embedding.unsqueeze(2)


    def forward(self,x):
        x = x + self.positional_embedding.unsqueeze(0)
        return self.dropout(x)       
        
        

        