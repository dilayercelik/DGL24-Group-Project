import torch.nn as nn
import torch


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, outpad=0):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, output_padding=outpad), #EDITED
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip[:, :, :x.shape[2], :x.shape[3]]), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=512, n_classes=10, three_layers=False): #changed n_feat to 512 (was 256)
        super(ContextUnet, self).__init__()

        self.three_layers = three_layers
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.out = nn.Sequential(
                nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
                nn.GroupNorm(8, n_feat),
                nn.ReLU(),
                nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
            )

        if not three_layers:
            self.to_vec = nn.Sequential(nn.AvgPool2d(67), nn.GELU())
            self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 67, 67),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
            )
            self.up1 = UnetUp(4 * n_feat, n_feat)
            self.up2 = UnetUp(2 * n_feat, n_feat)
        else:
            self.to_vec = nn.Sequential(nn.AvgPool2d(33), nn.GELU()) #EDITED Moritz (was 40) (33 if down3 included)
            self.up0 = nn.Sequential(
                # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
                nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 33, 33),  # otherwise just have 2*n_feat #EDITED Moritz (was 40) (also was 2 * n_feat)
                nn.GroupNorm(8, 4 * n_feat), #EDITED
                nn.ReLU(),
            )
            self.contextembed0 = EmbedFC(n_classes, 4 * n_feat) #ADDED
            self.timeembed0 = EmbedFC(1, 4 * n_feat) #ADDED
            self.down3 = UnetDown(2 * n_feat, 4 * n_feat) #ADDED
            self.up1 = UnetUp(8 * n_feat, 2 * n_feat, outpad=1) #EDITED (not sure why)
            self.up2 = UnetUp(4 * n_feat, n_feat)
            self.up3 = UnetUp(2 * n_feat, n_feat)


    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        if self.three_layers:
            x = self.init_conv(x)
            down1 = self.down1(x)
            down2 = self.down2(down1)
            down3 = self.down3(down2) #ADDED
            hiddenvec = self.to_vec(down3)

            # convert context to one hot embedding
            c_batch = c.shape[0]
            c = c.view(c_batch, -1)
            c_feats = c.shape[1]
            # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

            # mask out context if context_mask == 1
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1, c_feats)
            context_mask = 1 - context_mask  # need to flip 0 <-> 1
            c = c * context_mask

            # embed context, time step
            cemb0 = self.contextembed0(c).view(-1, self.n_feat * 4, 1, 1) #ADDED
            temb0 = self.timeembed0(t).view(-1, self.n_feat * 4, 1, 1) #ADDED
            cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
            temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
            temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

            # could concatenate the context embedding here instead of adaGN
            # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

            up1 = self.up0(hiddenvec)
            # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
            up2 = self.up1(cemb0 * up1 + temb0, down3)
            up3 = self.up2(cemb1 * up2 + temb1, down2)  # add and multiply embeddings
            up4 = self.up3(cemb2 * up3 + temb2, down1)
            out = self.out(torch.cat((up4, x), 1))
        else:
            x = self.init_conv(x)
            down1 = self.down1(x)
            down2 = self.down2(down1)
            hiddenvec = self.to_vec(down2)

            # convert context to one hot embedding
            c_batch = c.shape[0]
            c = c.view(c_batch, -1)
            c_feats = c.shape[1]
            # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

            # mask out context if context_mask == 1
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1, c_feats)
            context_mask = 1 - context_mask  # need to flip 0 <-> 1
            c = c * context_mask

            # embed context, time step
            cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
            temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
            temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

            # could concatenate the context embedding here instead of adaGN
            # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

            up1 = self.up0(hiddenvec)
            # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
            up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
            up3 = self.up2(cemb2 * up2 + temb2, down1)
            out = self.out(torch.cat((up3, x), 1))
        return out

class GCNContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=512, n_classes=10, three_layers=False): #changed n_feat to 512 (was 256)
        super(ContextUnet, self).__init__()

        self.three_layers = three_layers
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.out = nn.Sequential(
                nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
                nn.GroupNorm(8, n_feat),
                nn.ReLU(),
                nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
            )

        if not three_layers:
            self.to_vec = nn.Sequential(nn.AvgPool2d(67), nn.GELU())
            self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 67, 67),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
            )
            self.up1 = UnetUp(4 * n_feat, n_feat)
            self.up2 = UnetUp(2 * n_feat, n_feat)
        else:
            self.to_vec = nn.Sequential(nn.AvgPool2d(33), nn.GELU()) #EDITED Moritz (was 40) (33 if down3 included)
            self.up0 = nn.Sequential(
                # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
                nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 33, 33),  # otherwise just have 2*n_feat #EDITED Moritz (was 40) (also was 2 * n_feat)
                nn.GroupNorm(8, 4 * n_feat), #EDITED
                nn.ReLU(),
            )
            self.contextembed0 = EmbedFC(n_classes, 4 * n_feat) #ADDED
            self.timeembed0 = EmbedFC(1, 4 * n_feat) #ADDED
            self.down3 = UnetDown(2 * n_feat, 4 * n_feat) #ADDED
            self.up1 = UnetUp(8 * n_feat, 2 * n_feat, outpad=1) #EDITED (not sure why)
            self.up2 = UnetUp(4 * n_feat, n_feat)
            self.up3 = UnetUp(2 * n_feat, n_feat)


    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        if self.three_layers:
            x = self.init_conv(x)
            down1 = self.down1(x)
            down2 = self.down2(down1)
            down3 = self.down3(down2) #ADDED
            hiddenvec = self.to_vec(down3)

            # convert context to one hot embedding
            c_batch = c.shape[0]
            c = c.view(c_batch, -1)
            c_feats = c.shape[1]
            # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

            # mask out context if context_mask == 1
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1, c_feats)
            context_mask = 1 - context_mask  # need to flip 0 <-> 1
            c = c * context_mask

            # embed context, time step
            cemb0 = self.contextembed0(c).view(-1, self.n_feat * 4, 1, 1) #ADDED
            temb0 = self.timeembed0(t).view(-1, self.n_feat * 4, 1, 1) #ADDED
            cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
            temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
            temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

            # could concatenate the context embedding here instead of adaGN
            # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

            up1 = self.up0(hiddenvec)
            # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
            up2 = self.up1(cemb0 * up1 + temb0, down3)
            up3 = self.up2(cemb1 * up2 + temb1, down2)  # add and multiply embeddings
            up4 = self.up3(cemb2 * up3 + temb2, down1)
            out = self.out(torch.cat((up4, x), 1))
        else:
            x = self.init_conv(x)
            down1 = self.down1(x)
            down2 = self.down2(down1)
            hiddenvec = self.to_vec(down2)

            # convert context to one hot embedding
            c_batch = c.shape[0]
            c = c.view(c_batch, -1)
            c_feats = c.shape[1]
            # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

            # mask out context if context_mask == 1
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1, c_feats)
            context_mask = 1 - context_mask  # need to flip 0 <-> 1
            c = c * context_mask

            # embed context, time step
            cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
            temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
            temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

            # could concatenate the context embedding here instead of adaGN
            # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

            up1 = self.up0(hiddenvec)
            # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
            up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
            up3 = self.up2(cemb2 * up2 + temb2, down1)
            out = self.out(torch.cat((up3, x), 1))
        return out