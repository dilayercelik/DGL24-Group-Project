import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from AGSRNet_source.preprocessing import *
from AGSRNet_source.model import *
import torch.optim as optim

criterion = nn.MSELoss()
criterion_test = nn.L1Loss()


def train(model, subjects_adj, subjects_labels, args):

    bce_loss = nn.BCELoss()
    netD = Discriminator(args)
    print(netD)
    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    all_epochs_loss = []
    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            for lr, hr in zip(subjects_adj, subjects_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                
                padded_hr = pad_HR_adj(hr, args.padding)
                #lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(padded_hr).type(torch.FloatTensor)

                # NOTE: torch.symeig was deprecated in torch version 1.9
                # eig_val_hr, U_hr = torch.symeig(
                #     padded_hr, eigenvectors=True, upper=True)
                eig_val_hr, U_hr = eigenvalues, eigenvectors = torch.linalg.eigh(padded_hr, UPLO='U')

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)
                
                real_data = model_outputs.detach()
                
                model_outputs = unpad(model_outputs, args.padding)

                mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                    model.layer.weights, U_hr) + criterion(model_outputs, hr)

                error = criterion(model_outputs, hr)
                #real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)

                dc_loss_real = bce_loss(d_real, torch.ones(args.hr_dim, 1))
                dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
                dc_loss = dc_loss_real + dc_loss_fake

                dc_loss.backward()
                optimizerD.step()

                d_fake = netD(gaussian_noise_layer(padded_hr, args))

                gen_loss = bce_loss(d_fake, torch.ones(args.hr_dim, 1))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error)*100, "%")
            all_epochs_loss.append(np.mean(epoch_loss))


def test(model, test_adj, test_labels, args):

    g_t = []
    test_error = []
    test_error_mae = []
    preds_list = []

    # i = 0

    for lr, hr in zip(test_adj, test_labels):

        # all_zeros_lr = not np.any(lr)
        # all_zeros_hr = not np.any(hr)
        all_zeros_lr = torch.all(lr == 0)
        all_zeros_hr = torch.all(hr == 0)

        if all_zeros_lr == False and all_zeros_hr == False:
            #lr = torch.from_numpy(lr).type(torch.FloatTensor)
            #np.fill_diagonal(hr, 1)
            hr.fill_diagonal_(1)
            #hr = pad_HR_adj(hr, args.padding)
            #hr = torch.from_numpy(hr).type(torch.FloatTensor)
            preds, a, b, c = model(lr, args.lr_dim, args.hr_dim)
            preds = unpad(preds, args.padding)
            #preds = unpad(preds, args.padding)

            # if i == 0:
            #     print("Hr", hr)
            #     print("Preds  ", preds)
            #     plt.imshow(hr, origin='lower',  extent=[
            #         0, 10000, 0, 10], aspect=1000)
            #     plt.show(block=False)
            #     plt.imshow(preds.detach(), origin='lower',
            #                extent=[0, 10000, 0, 10], aspect=1000)
            #     plt.show(block=False)
            #     plt.imshow(hr - preds.detach(), origin='lower',
            #                extent=[0, 10000, 0, 10], aspect=1000)
            #     plt.show(block=False)

            # preds_list.append(preds.flatten().detach().numpy())
            preds_list.append(preds.flatten().detach())
            error = criterion(preds, hr)
            error_mae = criterion_test(preds, hr)
            g_t.append(hr.flatten())
            print(error.item())
            test_error.append(error.item())
            test_error_mae.append(error_mae.item())
            # i += 1

    print("Test error MSE: ", np.mean(test_error))
    print("Test error MAE: ", np.mean(test_error_mae))
    print()
    # preds_list = [val for sublist in preds_list for val in sublist]
    # g_t_list = [val for sublist in g_t for val in sublist]
    # binwidth = 0.01
    # bins = np.arange(0, 1 + binwidth, binwidth)
    # plt.hist(preds_list, bins=bins, range=(0, 1),
    #         alpha=0.5, rwidth=0.9, label='predictions')
    # plt.hist(g_t_list, bins=bins, range=(0, 1),
    #         alpha=0.5, rwidth=0.9, label='ground truth')
    # plt.xlim(xmin=0, xmax=1)
    # plt.legend(loc='upper right')
    # plt.title('GSR-UNet with self reconstruction: Histogram')
    # plt.show(block=False)
    # plt.plot(all_epochs_loss)
    # plt.title('GSR-UNet with self reconstruction: Loss')
    # plt.show()
