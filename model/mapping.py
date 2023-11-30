import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_layer_size=[16,14,12], latent_dim=16,output_size=128):
        super(EncoderDecoder, self).__init__()

        # Encoder
        encoder_layers = []
        num_hidden_layers = len(hidden_layer_size)
        flat_input_size = input_size[0] * input_size[1] * input_size[2]
        encoder_layers.append(nn.Linear(flat_input_size, hidden_layer_size[0]))
        encoder_layers.append(nn.ReLU())
        for i in range(1,num_hidden_layers):
            encoder_layers.append(nn.Linear(hidden_layer_size[i-1], hidden_layer_size[i]))
            encoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Linear(hidden_layer_size[-1], latent_dim))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
            nn.Tanh()  # Assuming you want values in the range [0, 1]
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input for the fully connected layer
        x = self.encoder(x)
        x = x.view(x.size(0), -1, 1, 1)  # Reshape for the convolutional layers
        print(x.shape)
        x = self.decoder(x)
        return x

