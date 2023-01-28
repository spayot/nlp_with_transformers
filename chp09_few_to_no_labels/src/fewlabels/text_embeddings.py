import torch
import transformers as tfm


class TransformerWithMeanPooling(torch.nn.Module):
    def __init__(self, model: tfm.AutoModel):
        super().__init__()
        self.model = model

    def forward(self, **inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        model_output = self.model(**inputs)
        return self.mean_pooling(
            model_output.last_hidden_state, inputs["attention_mask"]
        )

    def mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = self._expand_mask_over_emb_dim(
            attention_mask, dim=last_hidden_state.size()
        ).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def _expand_mask_over_emb_dim(
        self, attention_mask: torch.Tensor, dim: int
    ) -> torch.Tensor:
        return attention_mask.unsqueeze(-1).expand(dim)
