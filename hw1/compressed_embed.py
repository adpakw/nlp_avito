import torch
import torch.nn as nn


class CompressedWordEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        compression_rank: int,
        pretrained_weights: torch.Tensor,
    ) -> None:
        super(CompressedWordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.compression_rank = compression_rank

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(pretrained_weights, full_matrices=False)
            U_new = U[:, : self.compression_rank]
            S_new = S[: self.compression_rank]
            Vh_new = Vh[: self.compression_rank, :]

            self.embeddings_VE = nn.Embedding(self.vocab_size, self.compression_rank)
            self.linear_EH = nn.Linear(
                self.compression_rank, self.embedding_dim, bias=False
            )

            self.embeddings_VE.weight.copy_(U_new)
            self.linear_EH.weight.copy_(Vh_new.T * S_new.unsqueeze(0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings_E = self.embeddings_VE(token_ids)
        embeddings_H = self.linear_EH(embeddings_E)
        return embeddings_H
