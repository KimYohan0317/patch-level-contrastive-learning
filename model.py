import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import ViTModel

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads=8, ff_dim=512, dropout=0.1, device='cpu'):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, ff_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, input_size, ff_dim))
        encoder_layers = TransformerEncoderLayer(d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.layer_norm = nn.LayerNorm(ff_dim)
        self.output_layer = nn.Linear(ff_dim, output_size)
        self.device = device

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.layer_norm(x)
        x = self.transformer_encoder(x) + x
        x = self.output_layer(x)
        return x



class ViTTransformerModel(nn.Module):
    def __init__(self, input_shape ,num_classes, pooling_method, out_dim, transformer_layers=4, transformer_heads=8,  dropout=0.1,device='cpu'):
        super(ViTTransformerModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.vit.pooler.dense = nn.Linear(in_features=self.vit.config.hidden_size, out_features=out_dim, bias=True)
        self.transformer = TransformerModel(input_size=input_shape, output_size=out_dim, num_layers=transformer_layers, ff_dim=out_dim, num_heads=transformer_heads, dropout=dropout, device=device)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, 512),  # hidden layer 추가
            nn.ReLU(),
            nn.Dropout(0.3),  # 과적합 방지
            nn.Linear(512, num_classes))
        self.pooling_method = pooling_method

    def generate_indices(self, input_length):
        step = input_length // 24
        indices = []
        for i in range(1, 25):
            for j in range(1, 25):
                indices_i = [step * i - (step - k) for k in range(step)]
                indices_j = [step * j - (step - k) for k in range(step)]
                indices.append(indices_i + indices_j)
        return indices

    def forward(self, time_series, images):
        vit_output = self.vit(images)
        vit_cls_token = self.vit.pooler.dense(vit_output.last_hidden_state[:, 0, :])
        trf_tokens = self.transformer(time_series)[:, -1, :]
        patch_embeddings = vit_output.last_hidden_state[:, 1:, :]  # Exclude the [CLS] token

        token_embeddings = self.transformer(time_series)  # Using Transformer instead of BERT
        input_length = token_embeddings.size(1)
        all_tokens = []
        indices_list = self.generate_indices(input_length)

        for indices in indices_list:
            selected_tokens = token_embeddings[:, indices, :]
            if self.pooling_method == 'average':
                average_pooled = selected_tokens.mean(dim=1)  # average pooling
                all_tokens.append(average_pooled)

        token_embeddings = torch.stack(all_tokens, dim=1)  # Shape: (batch_size, 576, hidden_dim)
        hybrid_feature = torch.cat([trf_tokens, vit_cls_token], dim=-1)
        logits = self.classifier(hybrid_feature)
        return logits, patch_embeddings, token_embeddings, trf_tokens