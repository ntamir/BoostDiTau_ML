import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_n_pix, in_n_channels=3, filter_1_size=5, filter_2_size=5, filter_3_size=3, DOfrac=0.1):
        super().__init__()
      
        final_pix_size = int( (((in_n_pix-filter_1_size + 1)/2 - filter_2_size + 1)/2 - filter_3_size + 1)/2 )
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_n_channels, 16, filter_1_size),nn.PReLU(),nn.Dropout(DOfrac),nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, filter_2_size),nn.PReLU(),nn.Dropout(DOfrac),nn.MaxPool2d(2,2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, filter_3_size),nn.PReLU(),nn.Dropout(DOfrac),nn.MaxPool2d(2,2))
        self.classifier = nn.Sequential(nn.Linear(64*final_pix_size**2, 32),nn.PReLU(),nn.Dropout(DOfrac),
                                        nn.Linear(32,16),nn.PReLU(),nn.Dropout(DOfrac),
                                        nn.Linear(16, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1) 
        x = self.classifier(x)
        return x
