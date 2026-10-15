#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from eipl.utils import LossScheduler, tensor2numpy


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                #'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],
                "test_loss": loss[1],
            },
            savename,
        )

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss, total_img_loss, total_joint_loss, total_pt_loss = 0.0, 0., 0., 0.
        total_mask_loss, total_pose_loss = 0., 0.
        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):
            if "cpu" in self.device:
                x_img = x_img.to(self.device)
                y_img = y_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)

            state = None
            yi_list, yv_list = [], []
            dec_pts_list, enc_pts_list = [], []
            mask_list, pose_list = [], []
            self.optimizer.zero_grad(set_to_none=True)
            for t in range(x_img.shape[1] - 1):
                ret_tuple = self.model(
                    x_img[:, t], x_joint[:, t], state
                )
                if len(ret_tuple) == 6:
                    if self.model.pos_as_mask:
                        _yi_hat, _yv_hat, enc_ij, dec_ij, state, _yi_mask = ret_tuple
                        mask_list.append(_yi_mask)
                    else: 
                        _yi_hat, _yv_hat, enc_ij, dec_ij, state, _yi_pose = ret_tuple
                        pose_list.append(_yi_pose)
                elif len(ret_tuple) == 7:
                        _yi_hat, _yv_hat, enc_ij, dec_ij, state, _yi_mask, _yi_pose = ret_tuple
                        mask_list.append(_yi_mask)
                        pose_list.append(_yi_pose)
                else:
                    _yi_hat, _yv_hat, enc_ij, dec_ij, state = ret_tuple

                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)

            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))

            if self.model.pos_as_mask:
                tgt_masks = y_img[:,:,-1,...] # we concated the mask as the last channel to every rgb image, these masks t+1 are our tgts
                y_img = y_img[:,:,:-1,...]
                mask_logits = torch.stack(mask_list)
                mask_loss = nn.BCEWithLogitsLoss()(mask_logits, tgt_masks[:,1:,...].permute(1,0,2,3))

            if self.model.pos_as_state:
                poses = torch.stack(pose_list)
                pred_trans = poses[..., :3]
                pred_q = poses[..., 3:]
                tgt_trans = y_joint[..., -7:-4]
                tgt_q = y_joint[..., -4:]
                y_joint = y_joint[...,:-7]

                trans_loss = nn.MSELoss()(pred_trans, tgt_trans[:,1:,...].permute(1,0,2))

                # force the pred q to be unit quaternion with norm
                pred_q = F.normalize(pred_q, p=2, dim=-1, eps=1e-8)
                # chordal rotation loss 
                # alternatively try geodesic 
                # dot = dot.clamp(0.0, 1.0 - 1e-7)
                # angle = 2.0 * torch.acos(dot)
                # q_loss = angle.square().mean() 
                # dot = dot.abs()
                dot = torch.sum(pred_q * tgt_q[:,1:,...].permute(1,0,2), dim=-1)
                q_loss = (1.0 - dot.square()).mean()
                pose_loss = trans_loss + q_loss

            img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]
            joint_loss = nn.MSELoss()(yv_hat, y_joint[:, 1:]) * self.loss_weights[1]
            # Gradually change the loss value using the LossScheluder class.
            pt_loss = nn.MSELoss()(
                torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
            ) * self.scheduler(self.loss_weights[2])

            loss = img_loss + joint_loss + pt_loss
            if self.model.pos_as_mask:
                loss += mask_loss
                total_mask_loss += tensor2numpy(mask_loss)
            if self.model.pos_as_state:
                loss += pose_loss 
                total_pose_loss += tensor2numpy(pose_loss)

            total_loss += tensor2numpy(loss)
            total_img_loss += tensor2numpy(img_loss) 
            total_joint_loss += tensor2numpy(joint_loss)
            total_pt_loss += tensor2numpy(pt_loss)

            if training:
                loss.backward()
                self.optimizer.step()

        loss_dict = {
            "total_loss": total_loss / (n_batch + 1),
            "image_loss": total_img_loss / (n_batch + 1) ,
            "joint_loss": total_joint_loss / (n_batch + 1),
            "point_loss": total_pt_loss / (n_batch + 1) ,
        }
        if self.model.pos_as_mask:
            loss_dict["mask_loss"] = total_mask_loss / (n_batch + 1)
        if self.model.pos_as_state:
            loss_dict["pose_loss"] = total_pose_loss / (n_batch + 1)
        
        return loss_dict 