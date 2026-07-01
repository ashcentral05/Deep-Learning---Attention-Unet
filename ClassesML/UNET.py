import torch
import torch.nn as nn

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
        self.activation = self.hyperparameters["activation"]

        self.batch_normalization = self.hyperparameters["batch_normalization"]
        self.dropout_rate = self.hyperparameters["dropout_rate"]
        self.n_conv_layers = len(self.filters)

    def create_encoder(self):

        self.encoder_layers = nn.ModuleList()

        layer = DoubleConvBlock(
            in_channels=1,
            out_channels=self.filters[0],
            kernel_size=self.kernel_size,
            batch_normalization=self.batch_normalization,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate,
        )
        self.encoder_layers.append(layer)

        layer = MaxPoolingBlock(kernel_size=(2, 2))
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

            layer = MaxPoolingBlock(kernel_size=(2, 2))
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
        self.attention_layers = nn.ModuleList() 
    
    
        for i in range(self.n_conv_layers - 1, -1, -1):
        
           
            in_up = self.filters[i] * 2 if i == self.n_conv_layers - 1 else self.filters[i] #this condition is used to handle the bottleneck which starts at 1024
            out_up = self.filters[i - 1] 
            
            up_layer = UpConv2DBlock(
                in_channels=in_up,
                out_channels=out_up,
                kernel_size=self.kernel_size,
                activation=Utilities.get_activation(self.activation),
                batch_normalization=False,
                dropout_rate=self.dropout_rate
            )
            self.decoder_layers.append(up_layer)
            
          
            att_layer = AttentionBlock(F_g=out_up, F_l=out_up, F_int=out_up // 2)#vector g is from the decoding part and vector x is from the encoder (skip connection)
            self.attention_layers.append(att_layer)
            
            # 3. Le DoubleConvBlock après Concaténation
            # L'entrée reçois : l'Upconv (out_up) + la skip connection (out_up) -> donc out_up * 2 !
            in_double_conv = out_up * 2 
            out_double_conv = out_up     # La sortie finale de cet étage avant de monter au suivant
            
            conv_layer = DoubleConvBlock(
                in_channels=in_double_conv,
                out_channels=out_double_conv,
                kernel_size=self.kernel_size,
                batch_normalization=self.batch_normalization,
                activation=Utilities.get_activation(self.activation),
                dropout_rate=self.dropout_rate,
            )
            self.decoder_layers.append(conv_layer)
        
        #one last 1D convolution
        layer = Conv2DBlock(in_channels=self.filters[0],out_channels=1,activation=Utilities.get_activation('sigmoid'),kernel_size=1,batch_normalization=False,dropout_rate=0.0) 
        
        self.decoder_layers.append(layer)

    def forward(self, x):
        skip_connections = []
        
        for layer in self.encoder_layers:#during the encoder part we need to store the output of the convolution layers for the attention blocks
            if isinstance(layer, DoubleConvBlock):
                x = layer(x)
                skip_connections.append(x)
            elif isinstance(layer, MaxPoolingBlock):
                x = layer(x)
                
        x = self.bottle_neck_layer(x)
        
        skip_connections = skip_connections[::-1] #We inverse the indexes to get the latest added output first (the output at the lowest level)
        
        decoder_idx = 0 #this index is used to identify where are the up_layers (even index)
        for i in range(len(self.attention_layers)):
            up_layer = self.decoder_layers[decoder_idx]
            x = up_layer(x)
            
            skip = skip_connections[i]
            
            attention_layer = self.attention_layers[i]
            skip_attributed = attention_layer(g=x, x=skip) 
            
            x = torch.cat([x, skip_attributed], dim=1) #we concatenate the output of the attention block with the current value of x with respect to the number of channels
            
            conv_layer = self.decoder_layers[decoder_idx + 1]
            x = conv_layer(x)
            
            decoder_idx += 2
            
        final_layer = self.decoder_layers[-1]
        output = final_layer(x)
        
        return output

