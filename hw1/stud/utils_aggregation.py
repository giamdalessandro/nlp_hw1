import numpy as np
import torch 

from torch import Tensor, LongTensor 
from torch.nn import Module, Embedding


def embedding_lookUp(pretrained_emb: np.ndarray):
    """
    Lookup table that matches a list of word indexes to their respective embedding tensors,
    creating a pytorch embedding module.
    """
    pretrained_emb = np.array(pretrained_emb, dtype=np.float64)
    num_embeddings = pretrained_emb.shape[0]
    embedding_dim  = pretrained_emb.shape[1]
    return Embedding(num_embeddings, embedding_dim).from_pretrained(Tensor(pretrained_emb))


class EmbAggregation(Module):
    """ TODO
    This module implemente a lookup table module to fetch pretrained embeddings, 
    and applies one of the different embedding aggregation functions.
    """
    def __init__(self, pretrained: list):
        """
        Args:
            - pretrained: numpy matrix representing the pretrained embeddings.
        """
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        super().__init__()
        self.training  = False
        self.aggr_type = "concat"
        self.embedding = embedding_lookUp(pretrained)

    def forward(self, x: tuple):
        # apply embedding layer
        #sentence_embs = self.embedding(torch.LongTensor(x))
        s1_emb = self.embedding(LongTensor(x[0]))
        s2_emb = self.embedding(LongTensor(x[1]))

        # apply aggregation function
        #aggregated = self.dummy_aggreggation(sentence_embs)
        aggregated = self.concat_aggregation(s1_emb, s2_emb)
        return aggregated


    def dummy_aggreggation(self, sentence_embeddings):
        # TODO: too dummy
        # compute the mean of the embeddings of the two sentences
        return torch.mean(sentence_embeddings, dim=0).float()

    def concat_aggregation(self, s1_emb, s2_emb):
        # compute the mean of the embeddings of each 
        # sentence and concatenates them
        m_1 = torch.mean(s1_emb, dim=0).float()
        m_2 = torch.mean(s2_emb, dim=0).float()
        return torch.cat([m_1,m_2]).float()

    def square_diff_aggregation(self, s1_emb, s2_emb):
        m_1 = torch.mean(s1_emb, dim=0).float()
        m_2 = torch.mean(s2_emb, dim=0).float()
        m_1_sqr = torch.mul(m_1,m_1)
        m_2_sqr = torch.mul(m_2,m_2)
        return torch.abs(torch.sub(m_2,m_1)).float()

    def cosine_similarity(v1: Tensor, v2: Tensor):
        num = torch.sum(v1 * v2)
        den = torch.linalg.norm(v1) * torch.linalg.norm(v2)
        return (num / den).item()