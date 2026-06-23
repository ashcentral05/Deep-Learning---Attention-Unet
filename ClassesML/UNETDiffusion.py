import torch.nn as nn
import torch
from ClassesML.Blocks import *
from Utilities.Utilities import Utilities


class UNETDiffusion(nn.Module):
    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.hyperparameters = hyperparameters

        self.input_dim = self.hyperparameters["input_dim"]
        self.output_dim = self.hyperparameters["output_dim"]
        self.filters = hyperparameters["filters"]
        self.kernel_size = hyperparameters["kernel_size"]

        self.embedding_dim = self.hyperparameters["embedding_dim"]
        self.d_model = self.hyperparameters["d_model"]
        self.activation = self.hyperparameters["activation"]

        self.batch_normalization = self.hyperparameters["batch_normalization"]
        self.dropout_rate = self.hyperparameters["dropout_rate"]
        self.T = hyperparameters["T"]
        self.n_conv_layers = len(self.filters)

    def create_encoder(self):
        # Time embedding layer
        self.T_embedding_layer = DiffusionPositionalEncoding(
            num_embeddings=self.T + 1, d_model=self.embedding_dim
        )

        # conditionnal layer
        self.c_embedding_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.output_dim, embedding_dim=self.embedding_dim
            )
        )

        self.first_conv = Conv2DBlock(
            in_channels=self.input_dim[0],
            out_channels=self.filters[0],
            kernel_size=self.kernel_size,
            activation=self.activation,
            batch_normalization=self.batch_normalization,
            dropout_rate=self.dropout_rate,
        )

        self.encoder_layers = nn.ModuleList()

        layer = DiffuserEncoderEmbeddingBlock(
            in_channels=self.filters[0],
            out_channels=self.filters[0],
            embedding_dim=self.embedding_dim,
            kernel_size=self.kernel_size,
            batch_normalization=self.batch_normalization,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate,
        )
        self.encoder_layers.append(layer)

        layer = nn.AvgPool2d(kernel_size=2)
        self.encoder_layers.append(layer)

        for i in range(0, self.n_conv_layers - 1):
            layer = DiffuserEncoderEmbeddingBlock(
                in_channels=self.filters[i],
                out_channels=self.filters[i + 1],
                embedding_dim=self.embedding_dim,
                kernel_size=self.kernel_size,
                batch_normalization=self.batch_normalization,
                activation=Utilities.get_activation(self.activation),
                dropout_rate=self.dropout_rate,
            )
            self.encoder_layers.append(layer)

            layer = nn.AvgPool2d(kernel_size=2)
            self.encoder_layers.append(layer)

        # bottleneck layer
        self.bottle_neck_layer = DiffuserEncoderEmbeddingBlock(
            in_channels=self.filters[-1],
            out_channels=self.filters[-1] * 2,
            embedding_dim=self.embedding_dim,
            kernel_size=self.kernel_size,
            batch_normalization=self.batch_normalization,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate,
        )

    def create_decoder(self):
        pass

    def forward(self, x, T, c, mask):
        x = self.first_conv(x)
        skip_connections = []
        t_emb = self.T_embedding_layer(x)

        if c is not None:
            c_emb = self.c_embedding_layer(x)
        else:
            c_emb = None

        for encoder_block in self.encoder_layers:
            if isinstance(encoder_block, DiffuserEncoderEmbeddingBlock):
                x = encoder_block(x, t_emb, c_emb, mask)
                skip_connections.append(x)
            else:
                x = encoder_block(x)
        # Bottleneck
        x = self.bottle_neck_layer(x, t_emb, c_emb, mask)
