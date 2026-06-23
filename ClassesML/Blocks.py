import torch
import torch.nn as nn
import numpy as np
import math


def _to_tuple_int(size):
    if isinstance(size, int):
        return (size,)
    if isinstance(size, (tuple, list)):
        return tuple(int(s) for s in size)
    raise TypeError(
        f"size must be an int, tuple, or list, but found type {type(size).__name__}"
    )


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):

        super(DenseBlock, self).__init__()

        self.linear_layer = nn.Linear(in_size, out_size)
        self.activation = activation

        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm1d(out_size)
        else:
            self.batch_norm_layer = None

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.linear_layer(x)
        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)

        return x


class FlattenDenseBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):

        super(FlattenDenseBlock, self).__init__()

        in_size_flatten = int(np.prod(in_size))
        out_size_flatten = int(np.prod(out_size))
        self.flatten_layer = nn.Flatten()
        self.dense_layer = DenseBlock(
            in_size=in_size_flatten,
            out_size=out_size_flatten,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )
        self.unflatten_layer = nn.Unflatten(
            dim=1, unflattened_size=_to_tuple_int(out_size)
        )

    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.unflatten_layer(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):

        super(Conv2DBlock, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )  # adding some 0s around the input image to keep the same size after convolution
        self.activation = activation
        self.batch_norm_layer = (
            nn.BatchNorm2d(out_channels) if batch_normalization else None
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.batch_norm_layer:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x
#######################################################################################################################################
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU(), batch_normalization=False, dropout_rate=0.1):
        super(DoubleConvBlock, self).__init__()

        # Première convolution : fait passer les canaux de in_channels à out_channels
        self.conv_1 = Conv2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate
        )

        # Deuxième convolution : reste à out_channels
        self.conv_2 = Conv2DBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class MaxPoolingBlock(nn.Module):

    def __init__(self,kernel_size,stride):
        super(MaxPoolingBlock,self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size,stride=stride)
    
    def forward(self,x):
        return self.max_pool(x)
    

class UpConv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,            
        stride=2,                
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):
        super(UpConv2DBlock, self).__init__()
        
        
        self.up_conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,            
        )
        
        self.activation = activation
        self.batch_norm_layer = (
            nn.BatchNorm2d(out_channels) if batch_normalization else None
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
       
        x = self.up_conv_layer(x)
        if self.batch_norm_layer:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x
######################################################################################
class BasicResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):

        super(BasicResNetBlock, self).__init__()

        self.conv_layer_1 = Conv2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )
        self.conv_layer_2 = Conv2DBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

        if out_channels == in_channels:
            self.shortcut_layer = nn.Identity()  # if the number of channels is the same, we can use the identity function as a shortcut
        else:
            self.shortcut_layer = Conv2DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation=activation,
                batch_normalization=batch_normalization,
                dropout_rate=dropout_rate,
            )
            # if the number of channels is different, we need to use a convolutional layer as a shortcut to match the dimensions
            # but it is heavier than the identity function
        self.activation = activation
        self.batch_norm_layer = (
            nn.BatchNorm2d(out_channels) if batch_normalization else None
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):

        residual = x
        residual = self.shortcut_layer(
            residual
        )  # this is the shortcut connection, it will be added to the output of the convolutional layers

        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)

        x = x + residual

        return x


class UnFlattenDenseBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):

        super(UnFlattenDenseBlock, self).__init__()
        self.dense_layer = DenseBlock(
            in_size=in_size,
            out_size=np.prod(out_size),
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

        self.unflatten_layer = nn.Unflatten(
            dim=1, unflattened_size=_to_tuple_int(out_size)
        )

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.unflatten_layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_embeddings, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(num_embeddings, d_model)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # we use sine for even dimensions and cosine for odd dimensions to create a unique positional encoding for each position in the sequence
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe
        )  # we register the positional encoding as a buffer, which means that it will not be updated during training but it will be saved and loaded with the model

    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)
        return x

    def plot_positional_encoding(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(self.pe.T, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title("Positional Encoding Visualization")
        plt.xlabel("Position Index")
        plt.ylabel("Embedding Dimension")
        plt.show()


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        expansion_factor: int = 2,
        activation=nn.ReLU(),
        dropout_rate=0.0,
    ):

        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.mha_layer = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.attention_weights = None
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm_layer1 = nn.LayerNorm(input_dim)
        self.norm_layer2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(
            dropout_rate
        )  # dropout after the multi-head attention layer
        self.dropout2 = nn.Dropout(dropout_rate)  # dropout after the feed-forward layer

    def forward(self, x):

        # Self-Attention
        attention_out, self.attention_weights = self.mha_layer(x, x, x)

        attention_out = x + attention_out
        out = self.norm_layer1(attention_out)
        out = self.dropout1(out)

        # MLP part
        ff_out = self.mlp(out)
        out = ff_out + out
        out = self.norm_layer2(out)
        out = self.dropout2(out)
        return out


class DiffusionPositionalEncoding(nn.Module):
    def __init__(self, num_embeddings, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(num_embeddings, d_model)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # we use sine for even dimensions and cosine for odd dimensions to create a unique positional encoding for each position in the sequence
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe
        )  # we register the positional encoding as a buffer, which means that it will not be updated during training but it will be saved and loaded with the model

    def forward(self, x):
        return self.pe[x]


class DiffuserEncoderEmbeddingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embedding_dim,
        kernel_size,
        batch_normalization=True,
        activation=nn.ReLU(),
        dropout_rate=0.0,
    ):

        super().__init__()
        self.conv1 = Conv2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            batch_normalization=batch_normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        self.conv2 = Conv2DBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            batch_normalization=batch_normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        self.shortcut_layers = nn.ModuleList()
        if in_channels != out_channels:
            self.shortcut = Conv2DBlock(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=(1, 1),
                batch_normalization=False,
                activation=None,
                dropout_rate=dropout_rate,
            )

        else:
            self.shortcut = nn.Identity()
        # embedding
        self.fc_t = DenseBlock(
            embedding_dim,
            out_channels,
            batch_normalization=False,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        self.fc_c = DenseBlock(
            embedding_dim,
            out_channels,
            batch_normalization=False,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        x,
        t_emb,
        c_emb: None or torch.Tensor = None,
        mask: None or torch.Tensor = None,
    ):
        residual = x
        x = self.conv1(x)
        t_emb_proj = self.fc_t(t_emb)
        t_emb_proj = t_emb_proj.view(x.size(0), -1, 1, 1)
        x = t_emb_proj + x
        if c_emb is not None:
            c_emb_proj = self.fc_c(c_emb) * mask[:, None]
            c_emb_proj = c_emb_proj.view(x.size(0), -1, 1, 1)
            x = c_emb_proj + x
        x = self.conv2(x)

        residual = self.shortcut(residual)
        x = x + residual
        return x


class DiffuserDecoderEmbeddingBlock(DiffuserEncoderEmbeddingBlock):
    pass
