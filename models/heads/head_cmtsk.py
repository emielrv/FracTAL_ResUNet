import torch
import torch.nn.functional as F

# Helper classification head, for a single layer output
from nn.activations.sigmoid_crisp import SigmoidCrisp
from nn.layers.conv2Dnormed import Conv2DNormed
from nn.pooling.psp_pooling import PSP_Pooling


class HeadSingle(torch.nn.Module):
    def __init__(self, nfilters, NClasses, depth=2, norm_type='BatchNorm', norm_groups=None):
        super().__init__()

        self.logits = torch.nn.Sequential()
        for _ in range(depth):
            self.logits.add_module('conv_normed',
                                   Conv2DNormed(channels=nfilters, kernel_size=(3, 3), padding=1, _norm_type=norm_type,
                                                norm_groups=norm_groups))
            self.logits.add_module('relu', torch.nn.ReLU())
        self.logits.add_module('conv_out', torch.nn.Conv2d(nfilters, NClasses, kernel_size=1, padding=0))

    def forward(self, input):
        return self.logits(input)


class Head_CMTSK_BC(torch.nn.Module):
    # BC: Balanced (features) Crisp (boundaries)
    def __init__(self, nfilters_init, NClasses, norm_type='BatchNorm', norm_groups=None):
        super().__init__()

        self.model_name = "Head_CMTSK_BC"

        self.nfilters = nfilters_init  # Initial number of filters
        self.NClasses = NClasses

        self.psp_2ndlast = PSP_Pooling(self.nfilters, _norm_type=norm_type, norm_groups=norm_groups)

        # bound logits
        self.bound_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups)
        self.bound_Equalizer = Conv2DNormed(channels=self.nfilters, kernel_size=1, _norm_type=norm_type,
                                            norm_groups=norm_groups)

        # distance logits -- deeper for better reconstruction
        self.distance_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups)
        self.dist_Equalizer = Conv2DNormed(channels=self.nfilters, kernel_size=1, _norm_type=norm_type,
                                           norm_groups=norm_groups)

        self.Comb_bound_dist = Conv2DNormed(channels=self.nfilters, kernel_size=1, _norm_type=norm_type,
                                            norm_groups=norm_groups)

        # Segmentation logits -- deeper for better reconstruction
        self.final_segm_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups)

        self.CrispSigm = SigmoidCrisp()

        # Last activation, customization for binary results
        if self.NClasses == 1:
            self.ChannelAct = torch.nn.Sigmoid()
        else:
            self.ChannelAct = torch.nn.Softmax(dim=1)

    def forward(self, UpConv4, conv1):

        # second last layer
        convl = torch.cat((conv1, UpConv4), dim=1)
        conv = self.psp_2ndlast(convl)
        conv = F.relu(conv)

        # logits

        # 1st find the distance map, skeleton-like, topology info
        dist = self.distance_logits(convl)  # do not use max pooling for distance
        dist = self.ChannelAct(dist)
        distEq = F.relu(self.dist_Equalizer(dist))  # makes nfilters equals to conv and convl

        # Then find boundaries
        bound = torch.cat((conv, distEq), dim=1)
        bound = self.bound_logits(bound)
        bound = self.CrispSigm(bound)  # Boundaries are not mutually exclusive
        boundEq = F.relu(self.bound_Equalizer(bound))

        # Now combine all predictions in a final segmentation mask
        # Balance first boundary and distance transform, with the features
        comb_bd = self.Comb_bound_dist(torch.cat((boundEq, distEq), dim=1))
        comb_bd = F.relu(comb_bd)

        all_layers = torch.cat((comb_bd, conv), dim=1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.ChannelAct(final_segm)

        return final_segm, bound, dist
