"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.layer_sizes = layer_sizes
        self.embedding_sharing = embedding_sharing
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.user_embedding = [ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)]
        self.item_embedding = [ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)]
        self.user_bias = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1)
        self.user_bias.reset_parameters()
        self.item_bias = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1)
        self.item_bias.reset_parameters()
        
        if not embedding_sharing:
            self.user_embedding.append(ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim))
            self.item_embedding.append(ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim))
        
        for e in self.user_embedding:
            e.reset_parameters()
        for e in self.item_embedding:
            e.reset_parameters()
    
        self.relu = nn.ReLU()
        self.hidden_layers = []
        for i in range(len(layer_sizes)-1):
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.output_layer = nn.Linear(layer_sizes[-1],1)
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        u_score = self.user_embedding[0](user_ids)
        q_score = self.item_embedding[0](item_ids)
        if self.embedding_sharing:
            u_pred = u_score
            q_pred = q_score
        else:
            u_pred = self.user_embedding[1](user_ids)
            q_pred = self.item_embedding[1](item_ids)
            
        predictions = torch.sum(u_pred*q_pred,1) + self.user_bias(user_ids) + self.item_bias(item_ids)
        score = torch.cat((u_score, q_score, u_score*q_score),1)
        
        for hl in self.hidden_layers:
            score = hl(score)
            score = self.relu(score)
        score = self.output_layer(score)
        score = self.relu(score)
        
        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
