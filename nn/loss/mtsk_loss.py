import torch

from nn.loss.ftnmt_loss import ftnmt_loss


class mtsk_loss(torch.nn.Module):
    def __init__(self, depth=0, NClasses=2):
        super(mtsk_loss, self).__init__()

        self.ftnmt = ftnmt_loss(depth=depth)
        self.skip = NClasses

    def forward(self, _prediction, _label):
        pred_segm = _prediction[0]
        pred_bound = _prediction[1]
        pred_dists = _prediction[2]

        label_segm = _label[:, :self.skip, :, :]
        label_bound = _label[:, self.skip:2 * self.skip, :, :]
        label_dists = _label[:, 2 * self.skip:, :, :]

        loss_segm = self.ftnmt(pred_segm, label_segm)
        loss_bound = self.ftnmt(pred_bound, label_bound)
        loss_dists = self.ftnmt(pred_dists, label_dists)

        return (loss_segm + loss_bound + loss_dists) / 3.0
