from sched import scheduler
import torch
import os
from torch import nn
import torch.nn.functional as F

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # This wasn't necessary as my dataloader transforms didn't restrict the range of the input in this way
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image

def get_gradients(test_model,test_model_params,loss_function,dataloader,verbose=False,N=1):
    confounding_label_classes = test_model_params['n_classes']
    n_branches = test_model_params['n_branches']

    device = test_model_params['device']
    test_model.zero_grad(set_to_none=True)
    
    all_grads = list()
    for _ in range(n_branches):
        all_grads.append(list())
    
    for sample_idx in range(N):
        input,label = next(iter(dataloader))
        confounding_labels = torch.ones((len(label),confounding_label_classes)).float()
        input,confounding_labels = input.to(device),confounding_labels.to(device)
        output = test_model(input)
        for branch_idx,branch_output in enumerate(output):
            grads=list()
            loss = loss_function(branch_output,confounding_labels)
            loss.backward(retain_graph=True)
            for name,p in test_model.named_parameters():
                if p.grad is not None:
                    grads.append(torch.norm(p.grad))
            
            grads_tensor = torch.stack(grads)
            all_grads[branch_idx].append(grads_tensor)

            if verbose == True:
                print('Input '+str(sample_idx+1)+':')
                print('Branch '+str(branch_idx+1)+':')
                print('total grad: ',torch.sum(grads_tensor))
                print('mean grad: ',torch.mean(grads_tensor))
        test_model.zero_grad(set_to_none=True)

    for branch_idx in range(n_branches):
        all_grads[branch_idx] = torch.dstack(all_grads[branch_idx]).flatten(start_dim=0,end_dim=1)

    return(all_grads)

class EmbeddingAutoEncoder(torch.nn.Module):
    def __init__(self,start_dim,latent_dim):
        super().__init__()

        middle_dim = (start_dim+latent_dim)/2
        self.hidden_dim_1 = int((start_dim+middle_dim)/2)
        self.hidden_dim_2 = int((middle_dim+latent_dim)/2)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(start_dim,self.hidden_dim_1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim_2, latent_dim)
        )
    
        self.decoder = torch.nn.Sequential(  
            torch.nn.Linear(latent_dim, self.hidden_dim_2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim_2, self.hidden_dim_1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim_1, start_dim),
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded


class VAE_Encoder(nn.Module):
      ''' 
      This the encoder part of VAE
      '''
      def __init__(self, input_dim, hidden_dim, z_dim):
          '''
          Args:
              input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
              hidden_dim: A integer indicating the size of hidden dimension.
              z_dim: A integer indicating the latent dimension.
          '''
          super().__init__()

          self.linear = nn.Linear(input_dim, hidden_dim)
          self.mu = nn.Linear(hidden_dim, z_dim)
          self.var = nn.Linear(hidden_dim, z_dim)

      def forward(self, x):
          # x is of shape [batch_size, input_dim]

          hidden = F.relu(self.linear(x))
          # hidden is of shape [batch_size, hidden_dim]
          z_mu = self.mu(hidden)
          # z_mu is of shape [batch_size, latent_dim]
          z_var = self.var(hidden)
          # z_var is of shape [batch_size, latent_dim]

          return z_mu, z_var

class VAE_Decoder(nn.Module):
    ''' 
    This the decoder part of VAE
    '''
    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted

class VAE(nn.Module):
    ''' 
    This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self,start_dim,latent_dim):
        super().__init__()

        hidden_dim = int((start_dim+latent_dim)/2)
        self.enc = VAE_Encoder(start_dim,hidden_dim,latent_dim)
        self.dec = VAE_Decoder(latent_dim,hidden_dim,start_dim)

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(z)
        return (z, z_mu, z_var), predicted

class get_k_nearest_neighbours(nn.Module):
    def __init__(self,train_data):
        super().__init__()
        norms = torch.norm(train_data,dim=0)
        self.normalised_set = train_data/norms
    def forward(self,test_sample,k):
        normalised_test_sample = test_sample/torch.norm(test_sample)
        distances = torch.norm(torch.abs(normalised_test_sample-self.normalised_set),dim=1)
        #no min k, so make negative and take the top values, then make positive again
        nearest_k = -1*torch.topk(-1*distances,k=k)
        return nearest_k
    
def get_nearest_k_func(normalised_train,test_sample,k):
        normalised_test_sample = test_sample/torch.norm(test_sample)
        distances = torch.norm(torch.abs(normalised_test_sample-normalised_train),dim=1)
        #no min k, so make negative and take the top values, then make positive again
        nearest_k,_ = torch.sort(distances)
        return nearest_k[:k]

