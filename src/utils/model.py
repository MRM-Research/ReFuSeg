from typing import Optional, Union, List
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def maximum(*args):
    stacked = torch.stack(args)
    result, _ = torch.max(stacked, dim=0)
    return result

                
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)
                         
class SegmentationModel(torch.nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder1.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
    
    def initialize_contrastive_head(self):
        head = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(in_features=self.encoder1.out_channels[-1], out_features=512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(in_features=512, out_features=64),
                            nn.BatchNorm1d(64),
                        )
        return head

    def forward(self, x1=None, x2=None, x3=None, x4=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x1)
        self.check_input_shape(x2)
        self.check_input_shape(x3)
        self.check_input_shape(x4)

        if x1 is not None:
            features1 = self.encoder1(x1)
            f1 = features1[-1]
        if x2 is not None:
            features2 = self.encoder2(x2)
            f2 = features2[-1]
        if x3 is not None:
            features3 = self.encoder3(x3)
            f3 = features3[-1]
        if x4 is not None:
            features4 = self.encoder4(x4)
            f4 = features4[-1]
        
        for ind in range(len(features1)):
            # features[ind] = (features[ind]+features1[ind])/2
            # features[ind] = features1[ind]
            features1[ind] = maximum(features1[ind], features2[ind], features3[ind], features4[ind])
            # features[ind] = torch.cat((features[ind],features1[ind]),1)
    
        decoder_output = self.decoder(*features1)

        masks = self.segmentation_head(decoder_output)

        if self.contrastive:
            f1= self.contrastive_head1(f1)
            f2= self.contrastive_head2(f2)
            f3= self.contrastive_head3(f3)
            f4= self.contrastive_head4(f4)
            return masks, f1,  f2, f3, f4
        return masks

    @torch.no_grad()
    def predict(self, x1=None, x2=None, x3=None, x4=None):
        if self.training:
            self.eval()
        if self.contrastive:
            x, _, _ , _, _= self.forward(x1, x2, x3, x4)
            return x
        # if y is not None:
        #     x = self.forward(x,y)
        #     return x
        else:
            x = self.forward(x1, x2, x3, x4)

        return x
                         
class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        contrastive: bool = False,
    ):
        super().__init__()
        self.contrastive = contrastive
        self.encoder1 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder2 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder3 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder4 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=(self.encoder1.out_channels),
            # encoder_channels=tuple([2*item for item in self.encoder.out_channels]),
            decoder_channels=decoder_channels,
            # decoder_channels=tuple([2*item for item in decoder_channels]),
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        if contrastive:
            self.contrastive_head1= self.initialize_contrastive_head()
            self.contrastive_head2= self.initialize_contrastive_head()
            self.contrastive_head3= self.initialize_contrastive_head()
            self.contrastive_head4= self.initialize_contrastive_head()

        self.name = "u-{}".format(encoder_name)
        self.initialize()
