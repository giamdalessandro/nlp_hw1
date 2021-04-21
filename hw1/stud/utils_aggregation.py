import numpy as np
import torch 

from torch import Tensor, LongTensor 
from torch.nn import Module, Embedding

def embedding_lookUp(pretrained_emb: np.ndarray):
    """
    Lookup table that matches a list of word indexes to their respective embedding tensors,
    creating a pytorch embedding module.
    """
    num_embeddings = pretrained_emb.shape[0]
    embedding_dim  = pretrained_emb.shape[1]
    return Embedding(num_embeddings, embedding_dim).from_pretrained(Tensor(pretrained_emb))


class EmbAggregation(Module):
    """ TODO
    This module implemente a lookup table module to fetch pretrained embeddings, 
    and applies one of the different embedding aggregation functions.
    """
    def __init__(self, pretrained: np.array):
        """
        Args:
            - pretrained: numpy matrix representing the pretrained embeddings.
        """
        super().__init__()
        self.training  = False
        self.embedding = embedding_lookUp(pretrained)


    def forward(self, x: tuple):
        #sentence_embs = self.embedding(torch.LongTensor(x))
        s1_emb = self.embedding(LongTensor(x[0]))
        s2_emb = self.embedding(LongTensor(x[1]))
        #aggregated = self.dummy_aggreggation(sentence_embs)
        aggregated = self.diff_aggregation(s1_emb, s2_emb)
        return aggregated

    def dummy_aggreggation(self, sentence_embeddings):
        # TODO: too dummy
        # compute the mean of the embeddings of a sentence
        return torch.mean(sentence_embeddings, dim=0).float()

    def diff_aggregation(self, s1_emb, s2_emb):
        m_1 = torch.mean(s1_emb, dim=0).float()
        m_2 = torch.mean(s2_emb, dim=0).float()
        return torch.cat([m_1,m_2]).float()

    def lemma_aggregation(self, s1_emb, s2_emb, lemma):
        return