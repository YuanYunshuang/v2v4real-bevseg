import numpy as np
import torch
import torch.nn as nn

from einops import rearrange


class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super(VanillaSegLoss, self).__init__()
        self.target = args['target']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        if self.target == 'dynamic' or self.target != 'static':
            self.d_weights = args['d_weights']
            self.d_coe = args['d_coe']
            self.loss_func_dynamic = \
                nn.CrossEntropyLoss(
                    weight=torch.Tensor([1., self.d_weights]).cuda())
        if self.target == 'static' or self.target != 'dynamic':
            self.s_weights = args['s_weights']
            self.s_coe = args['s_coe']
            self.loss_func_static = \
                nn.CrossEntropyLoss(
                    weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())

        self.loss_dict = {}

    def forward(self, output_dict, gt_dict):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """
        total_loss = 0
        if self.target == 'dynamic' or self.target != 'static':
            dynamic_pred = output_dict['dynamic_seg']
            # during training, we only need to compute the ego vehicle's gt loss
            dynamic_gt = gt_dict['gt_dynamic'].long()
            # dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')
            # dynamic_gt = torch.stack([1 - dynamic_gt, dynamic_gt], dim=1).float()  # to one hot
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')

            # import matplotlib.pyplot as plt
            # import numpy as np
            # img1 = dynamic_pred.softmax(1)[0, 1].detach().cpu().numpy()
            # img2 = dynamic_gt[0, 1].detach().cpu().numpy()
            # img = np.concatenate([img1, np.ones_like(img1[:10]) * 0.1, img2], axis=0)
            # plt.imshow(img)
            # plt.show()
            # plt.close()

            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            total_loss += self.d_coe * dynamic_loss
            self.loss_dict.update({'dynamic_loss': dynamic_loss})

        if self.target == 'static' or self.target != 'dynamic':
            static_pred = output_dict['static_seg']
            # during training, we only need to compute the ego vehicle's gt loss
            static_gt = gt_dict['gt_static']
            # static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)
            total_loss += self.s_coe * static_loss
            self.loss_dict.update({'static_loss': static_loss})

        self.loss_dict.update({'total_loss': total_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                " || Dynamic Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), static_loss.item(), dynamic_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                  " || Dynamic Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), static_loss.item(), dynamic_loss.item()))

        if writer is not None:
            writer.add_scalar('Static_loss', static_loss.item(),
                              epoch*batch_len + batch_id)
            writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                              epoch*batch_len + batch_id)




