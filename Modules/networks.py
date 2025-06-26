import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm

# Custom VGG19 model architecture
class CustomVGG19(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomVGG19, self).__init__()
        self.model = models.vgg19(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom ResNet152 model architecture
class CustomResNet152(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomResNet152, self).__init__()
        self.model = models.resnet152(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom Inception model architecture
class CustomInception(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomInception, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained)
        self.model.transform_input = False
        self.model.aux_logits = False
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom DenseNet121 model architecture
class CustomDenseNet121(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomDenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        self.model.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Two-stream VGG19 model architecture
class TwoStreamVGG(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamVGG, self).__init__()

        # Load pretrained vgg19
        self.vgg_nir = models.vgg19(pretrained=pretrained)
        self.vgg_rgb = models.vgg19(pretrained=pretrained)

        # Modify the nir model
        self.vgg_nir.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        num_nir_features =  self.vgg_nir.classifier[-1].in_features
        self.vgg_nir.classifier[-1] = nn.Linear(num_nir_features, 4096)

        # Modify the rgb model
        num_rgb_features =  self.vgg_rgb.classifier[-1].in_features
        self.vgg_rgb.classifier[-1] = nn.Linear(num_rgb_features, 4096)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.vgg_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.vgg_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output
    

# Two-stream ResNet152 model architecture
class TwoStreamResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamResNet, self).__init__()

        # Load pretrained resnet152
        self.resnet_nir = models.resnet152(pretrained=pretrained)
        self.resnet_rgb = models.resnet152(pretrained=pretrained)

        # Modify the nir model
        self.resnet_nir.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_nir_features = self.resnet_nir.fc.in_features
        self.resnet_nir.fc = nn.Linear(num_nir_features, 2048)

        # Modify the rgb model
        num_rgb_features = self.resnet_rgb.fc.in_features
        self.resnet_rgb.fc = nn.Linear(num_rgb_features, 2048)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(2048*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.resnet_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.resnet_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output
    

# Two-stream Inception model architecture
class TwoStreamInception(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamInception, self).__init__()

        # Load pretrained inception_v3
        self.inception_nir = models.inception_v3(pretrained=pretrained)
        self.inception_rgb = models.inception_v3(pretrained=pretrained)

        # Modify the nir model
        self.inception_nir.transform_input = False
        self.inception_nir.aux_logits = False 
        self.inception_nir.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        num_nir_features = self.inception_nir.fc.in_features
        self.inception_nir.fc = nn.Linear(num_nir_features, 2048)

        # Modify the rgb model
        self.inception_rgb.aux_logits = False 
        num_rgb_features = self.inception_rgb.fc.in_features
        self.inception_rgb.fc = nn.Linear(num_rgb_features, 2048)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(2048*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.inception_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.inception_rgb(rgb_input) 

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output
    
# Two-stream DenseNet121 model architecture
class TwoStreamDenseNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamDenseNet, self).__init__()

        # Load pretrained DenseNet121
        self.densenet_nir = models.densenet121(pretrained=pretrained)
        self.densenet_rgb = models.densenet121(pretrained=pretrained)

        # Modify the nir model
        self.densenet_nir.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_nir_features = self.densenet_nir.classifier.in_features
        self.densenet_nir.classifier = nn.Linear(num_nir_features, 1024)

        # Modify the rgb model
        num_rgb_features = self.densenet_rgb.classifier.in_features
        self.densenet_rgb.classifier = nn.Linear(num_rgb_features, 1024)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.densenet_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.densenet_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output

class PretrainedViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained,
            drop_rate=0.1,          # Dropout for regularization
            drop_path_rate=0.1      # Stochastic depth
        )
        
        # Modify for single-channel input while preserving pretrained weights
        if pretrained:
            weight = self.vit.patch_embed.proj.weight
            self.vit.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
            # Average the pretrained weights across RGB channels
            self.vit.patch_embed.proj.weight.data = weight.sum(dim=1, keepdim=True)
        else:
            self.vit.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        
        # Custom head for regression
        self.vit.head = nn.Sequential(
            nn.LayerNorm(self.vit.head.in_features),
            nn.Linear(self.vit.head.in_features, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        return self.vit(x)
    
