import torch.nn as nn
import torch
from ClassesML.Blocks import *
from Utilities.Utilities import Utilities

#DoubleConvBlock
class UNET(nn.Module):

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
        self.n_conv_layers = len(self.filters)

    def create_encoder(self):

        self.encoder_layers = nn.ModuleList()

        layer = DoubleConvBlock(
            in_channels=self.filters[0],
            out_channels=self.filters[0],
            kernel_size=self.kernel_size,
            batch_normalization=self.batch_normalization,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate,
        )
        self.encoder_layers.append(layer)

        layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.encoder_layers.append(layer)

        for i in range(0, self.n_conv_layers - 1):
            layer = DoubleConvBlock(
                in_channels=self.filters[i],
                out_channels=self.filters[i + 1],
                kernel_size=self.kernel_size,
                batch_normalization=self.batch_normalization,
                activation=Utilities.get_activation(self.activation),
                dropout_rate=self.dropout_rate,
            )
            self.encoder_layers.append(layer)

            layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.encoder_layers.append(layer)

        # bottleneck layer
        self.bottle_neck_layer = DoubleConvBlock(
            in_channels=self.filters[-1],
            out_channels=self.filters[-1] * 2,
            kernel_size=self.kernel_size,
            batch_normalization=self.batch_normalization,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate,
        )

    def create_decoder(self):
        
        self.decoder_layers = nn.ModuleList()

        for i in range(self.n_conv_layers - 1, 0, -1):

            layer = nn.Upsample((self.filters[i], self.filters[i]))
            self.decoder_layers.append(layer)


            layer = DoubleConvBlock(
                in_channels=self.filters[i],
                out_channels=self.filters[i - 1],
                kernel_size=self.kernel_size,
                batch_normalization=self.batch_normalization,
                activation=Utilities.get_activation(self.activation),
                dropout_rate=self.dropout_rate,
            )
            self.decoder_layers.append(layer)

            

    def forward(self, x):
        x = self.first_conv(x)
        skip_connections = []

        for encoder_block in self.encoder_layers:
            if isinstance(encoder_block, DiffuserEncoderEmbeddingBlock):
                x = encoder_block(x)
                skip_connections.append(x)
            else:
                x = encoder_block(x)
                
        # Bottleneck
        x = self.bottle_neck_layer(x)

        for i, decoder_block in enumerate(self.decoder_layers):
            x = torch.cat(skip_connections[i], x)
            x = decoder_block(x)


