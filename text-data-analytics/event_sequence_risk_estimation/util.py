import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class data_generator(Dataset):
    def __init__(self, subject, occurrence, phase, data_length,device):
        """
        Args:
        data_tensor: nsample * nT
        """
        self.device = device        
        self.subject = subject.to(device)
        self.occurrence = occurrence.to(device)
        self.phase = phase.to(device)
        self.data_length = torch.from_numpy(data_length).to(device)
    def __len__(self): 
        """
        Return the number of samles
        """
        return self.subject.shape[0]
    def __getitem__(self, idx):
        """
        Return the number of samles
        """        
        return self.subject[idx], self.occurrence[idx], self.phase[idx], self.data_length[idx]

    
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNModel(nn.Module):
    def __init__(self, phase_size, occurrence_size, subject_size, embedding_dim,  latent_dim, num_layer, batch_size, hidden_dim, device):
        super(RNNModel, self).__init__() 
        self.latent_dim = latent_dim
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.phase_size = phase_size
        self.occurrence_size = occurrence_size
        self.hidden_dim = hidden_dim
        
        self.subject_size = subject_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.word_embedding_phase = nn.Embedding(
            num_embeddings=self.phase_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.phase_size-1,
        ).to(self.device)

        self.word_embedding_occurrence = nn.Embedding(
            num_embeddings=self.occurrence_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.occurrence_size-1,
        ).to(self.device)

        self.word_embedding_subject = nn.Embedding(
            num_embeddings=self.subject_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.subject_size-1,
        ).to(self.device)
        
        self.output = nn.Linear(latent_dim, hidden_dim).to(device)
        
        self.rnn = nn.LSTM(     
            input_size=3 * self.embedding_dim,       
            hidden_size=self.latent_dim,     # class number 
            num_layers=self.num_layer,       # number of RNN layers
            batch_first=True,  
            bidirectional=False,
        ).to(self.device)
        
    def init_hidden(self):
        h0 = torch.zeros(self.num_layer, self.batch_size, self.latent_dim, requires_grad=True).to(self.device)
        c0 = torch.zeros(self.num_layer, self.batch_size, self.latent_dim, requires_grad=True).to(self.device)
        return (h0, c0)    
    
    def forward(self, X_phase, X_occurrence, X_subject, X_lengths, hidden):
        h_phase = self.word_embedding_phase(X_phase)
        h_occurrence = self.word_embedding_occurrence(X_occurrence)
        h_subject = self.word_embedding_subject(X_subject)
        h = torch.cat((h_phase,h_occurrence,h_subject),axis=1).unsqueeze(0)
        X, hidden = self.rnn(h, hidden)
        out = self.output(X[0])
        return out, hidden
    
    
class SequentialPrediction(nn.Module):
    def __init__(self,phase_size,occurrence_size,subject_size, embedding_dim, batch_size,hidden_dim, device):
        super(SequentialPrediction, self).__init__() 
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        self.phase_size = phase_size
        self.occurrence_size = occurrence_size
        self.subject_size = subject_size
                
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.word_embedding_phase = nn.Embedding(
            num_embeddings=self.phase_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.phase_size-1,
        ).to(self.device)

        self.word_embedding_occurrence = nn.Embedding(
            num_embeddings=self.occurrence_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.occurrence_size-1,
        ).to(self.device)

        self.word_embedding_subject = nn.Embedding(
            num_embeddings=self.subject_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.subject_size-1,
        ).to(self.device)
        self.output = nn.Linear(3*embedding_dim, hidden_dim).to(device)
    def forward(self, X_phase, X_occurrence, X_subject, X_lengths):
        h_phase = self.word_embedding_phase(X_phase)
        h_occurrence = self.word_embedding_occurrence(X_occurrence)
        h_subject = self.word_embedding_subject(X_subject)
        h = torch.cat((h_phase,h_occurrence,h_subject),axis=1)
        out = F.relu(self.output(F.relu(h)))

        return out
    
def train():
    hidden = model.init_hidden()
    for X_subject, X_occurrence, X_phase, X_length in training_generator_pred:
        optimizer.zero_grad()
        X, hidden = model(X_subject, X_length, hidden)
        seq_length = X.shape[1]
        X_subject = X_subject[:,1:seq_length+1].reshape(-1)
        X_occurrence = X_occurrence[:,1:seq_length+1].reshape(-1)
        X_phase = X_phase[:,1:seq_length+1].reshape(-1)
        proba = hierarchical_softmax(X.view(-1,X.shape[2]), X_length, X_subject, X_occurrence, X_phase)
        loss = -torch.mean(torch.log(proba))
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        print(loss.data)
        
class HierarchicalSoftmax(nn.Module):
    def __init__(self, hidden_dim, phase_to_occurrence_dict, occurrence_to_subj_dict,subj_to_occurrence_dict,device):
        super(HierarchicalSoftmax, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.phase_to_occurrence_dict = phase_to_occurrence_dict
        
        self.occurrence_to_subj_dict = occurrence_to_subj_dict
        self.subj_to_occurrence_dict = subj_to_occurrence_dict
        
        self.to_phase = nn.Linear(self.hidden_dim, len(self.phase_to_occurrence_dict))
        
        self.phase_to_occurrence = nn.ModuleDict({str(key): nn.Linear(hidden_dim,len(self.phase_to_occurrence_dict[key])) for key in self.phase_to_occurrence_dict.keys()})
        
        self.occurrence_to_subj = nn.ModuleDict({str(key): nn.Linear(hidden_dim,len(self.occurrence_to_subj_dict[key])) for key in self.occurrence_to_subj_dict.keys()})
   
        self.softmax1 = nn.Softmax(dim=1)
    
        self.softmax = nn.Softmax(dim=0)
    
        self.device = device
    
    def forward(self, X, X_length, X_subject, X_occurrence, X_phase):
        device = self.device
        phase_logit = self.to_phase(X)
        phase_proba = self.softmax1(phase_logit)
        phase_proba = phase_proba[torch.arange(phase_proba.shape[0]), X_phase].to(device)
        
        occurrence_proba = torch.zeros(len(X_occurrence)).to(self.device)
        for i in range(len(X_phase)):
            key = X_phase[i].cpu().numpy()
            buffer = self.phase_to_occurrence[str(key)](X[i])
            occurrence_proba[i] = self.softmax(buffer)[self.phase_to_occurrence_dict[int(key)].index(X_occurrence[i])]

        subject_proba = torch.zeros(len(X_subject)).to(self.device)
        for i in range(len(X_subject)):
            key = X_occurrence[i].cpu().numpy()
            buffer = self.occurrence_to_subj[str(key)](X[i])
            subject_proba[i] = self.softmax(buffer)[self.occurrence_to_subj_dict[int(key)].index(X_subject[i])]       
        target_proba = phase_proba*occurrence_proba*subject_proba
        return phase_proba,phase_proba*occurrence_proba,target_proba
    
    
    def predict(self, X, X_length, X_subject, X_occurrence, X_phase):
        phase_logit = self.to_phase(X)
        phase_proba = self.softmax1(phase_logit)
        nlength = X.shape[0]

        occurance_proba = torch.zeros(nlength,max(self.occurrence_to_subj_dict.keys())+1, requires_grad=True).to(self.device)
        subject_proba = torch.zeros(nlength,max(self.subj_to_occurrence_dict.keys())+1,requires_grad=True).to(self.device)
                
        for i_time in range(len(X_phase)):
            for i_phase in self.phase_to_occurrence_dict.keys():
                phase_to_occurance_prob = self.softmax(self.phase_to_occurrence[str(i_phase)](X[i_time]))
                occurance_proba[i_time,self.phase_to_occurrence_dict[i_phase]] += phase_proba[i_time,i_phase] * phase_to_occurance_prob               
            for i_occurrence in self.occurrence_to_subj_dict.keys():
                occurrence_to_subj_prob = self.softmax(self.occurrence_to_subj[str(i_occurrence)](X[i_time]))
                subject_proba[i_time,self.occurrence_to_subj_dict[i_occurrence]] += occurance_proba[i_time,i_occurrence]*occurrence_to_subj_prob
                
        return phase_proba, occurance_proba, subject_proba
            
            
def show_event(Occurrences_merge, ev_id, DataDict):
    table = Occurrences_merge[Occurrences_merge['ev_id']==ev_id]
    table.loc[:,'Occurrence_Code'] = return_meaning('Occurrence_Code', table['Occurrence_Code'], DataDict)
    table.loc[:,'Phase_of_Flight'] = return_meaning('Phase_of_Flight', table['Phase_of_Flight'], DataDict)
    table.loc[:,'Subj_Code'] = return_meaning('Subj_Code', table['Subj_Code'], DataDict)
    table.loc[:,'Modifier_Code'] = return_meaning('Modifier_Code', table['Modifier_Code'], DataDict)
    return table

def return_meaning(key, code_list, DataDict):
    refer = DataDict[DataDict['Column']==key].iloc[:,1:-2]
    code_mean = []
    for code in code_list:
        if str(int(code)) in list(refer['code_iaids']):
            index = int(np.where(refer['code_iaids']==str(int(code)))[0][0])
            #code_mean.append(list(refer['meaning'][index])[0])
            code_mean.append(refer.iloc[index,4])
        else:
            code_mean.append('unknown')
    return list(code_mean)

def create_risk_estimator(pred_phase, pred_occurrence, pred_subject):
    subject_buffer = np.zeros(len(subject_code_corpus))
    phase_buffer = np.zeros(len(phase_code_corpus))
    occurrence_buffer = np.zeros(len(occurrence_code_corpus))
    for j in pred_subject:
        subject_buffer[int(j)] = 1
    for j in pred_occurrence:
        occurrence_buffer[int(j)] = 1
    for j in pred_phase:
        phase_buffer[int(j)] = 1
    a = np.concatenate((subject_buffer,occurrence_buffer, phase_buffer),axis=0)
    return a