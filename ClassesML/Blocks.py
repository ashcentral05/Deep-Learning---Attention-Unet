import torch
import torch.nn as nn
import numpy as np
import math


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


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=nn.ReLU(),
        batch_normalization=False,
        dropout_rate=0.1,
    ):

        super(DoubleConvBlock, self).__init__()

        self.conv_1 = Conv2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

        self.conv_2 = Conv2DBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


class MaxPoolingBlock(nn.Module):
    def __init__(self, kernel_size, stride):

        super(MaxPoolingBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):

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


class AttentionBlock(nn.Module):
    def __init__(
        self,
        gate_channels,
        encoder_channels,
        intermediate_channels,
        activation=nn.ReLU,
        dropout_rate=0.1,
    ):

        super(AttentionBlock, self).__init__()

        self.project_gate = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.project_encoder = nn.Sequential(
            nn.Conv2d(
                encoder_channels,
                intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.compute_attention_weights = nn.Sequential(
            nn.Conv2d(
                intermediate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = activation(inplace=True)

    def forward(self, g, x):

        g1 = self.project_gate(g)
        x1 = self.project_encoder(x)
        psi = self.relu(g1 + x1)
        psi = self.compute_attention_weights(psi)

        return x * psi
