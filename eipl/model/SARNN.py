#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax


class SARNN(nn.Module):
    #:: SARNN
    """SARNN: Spatial Attention with Recurrent Neural Network.
    This model "explicitly" extracts positions from the image that are important to the task, such as the work object or arm position,
    and learns the time-series relationship between these positions and the robot's joint angles.
    The robot is able to generate robust motions in response to changes in object position and lighting.

    Arguments:
        rec_dim (int): The dimension of the recurrent state in the LSTM cell.
        k_dim (int, optional): The dimension of the attention points.
        joint_dim (int, optional): The dimension of the joint angles.
        temperature (float, optional): The temperature parameter for the softmax function.
        heatmap_size (float, optional): The size of the heatmap in the InverseSpatialSoftmax layer.
        kernel_size (int, optional): The size of the convolutional kernel.
        self.activation (str, optional): The name of self.activation function.
        im_size (list, optional): The size of the input image [height, width].
    """

    def __init__(
        self,
        rec_dim,
        k_dim=5,
        joint_dim=14,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        im_size=[64, 64],
        pos_as_mask=False,
        pos_as_state=False
    ):
        super(SARNN, self).__init__()

        self.k_dim = k_dim
        self.pos_as_mask = pos_as_mask
        self.pos_as_state = pos_as_state
        self.activation = nn.LeakyReLU(negative_slope=0.3)

        sub_im_size = [
            im_size[0] - 3 * (kernel_size - 1),
            im_size[1] - 3 * (kernel_size - 1),
        ]
        self.temperature = temperature
        self.heatmap_size = heatmap_size

        # increase input channels as pose is incorportated as concat channel
        if pos_as_mask:
            im_enc_inp_sz = 4
        else:
            im_enc_inp_sz = 3

        # Positional Encoder
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(im_enc_inp_sz, 16, 3, 1, 0),  # Convolutional layer 1
            self.activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            self.activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            self.activation,
            SpatialSoftmax(
                width=sub_im_size[0],
                height=sub_im_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )
       
        # Image Encoder
        self.im_encoder = nn.Sequential(
            nn.Conv2d(im_enc_inp_sz, 16, 3, 1, 0),  # Convolutional layer 1
            self.activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            self.activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            self.activation,
        )

        rec_in = joint_dim + self.k_dim * 2
        if self.pos_as_state:
            rec_in += 7 # 3 translation + 4 rot as quaternion (wxyz)
        self.rec = nn.LSTMCell(rec_in, rec_dim)  # LSTM cell

        # Joint Decoder
        self.decoder_joint = nn.Sequential(
            nn.Linear(rec_dim, joint_dim), self.activation
        )  # Linear layer and self.activation

        # Point Decoder
        self.decoder_point = nn.Sequential(
            nn.Linear(rec_dim, self.k_dim * 2), self.activation
        )  # Linear layer and self.activation

        # Pose Decoder
        if self.pos_as_state:
            self.decoder_pose = nn.Sequential(
                nn.Linear(rec_dim, 1), self.activation
            )  # Linear layer and self.activation

        # Inverse Spatial Softmax
        self.issm = InverseSpatialSoftmax(
            width=sub_im_size[0],
            height=sub_im_size[1],
            heatmap_size=self.heatmap_size,
            normalized=True,
        )

        # Image Decoder
        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(
                self.k_dim, 32, 3, 1, 0
            ),  # Transposed Convolutional layer 1
            self.activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            self.activation,
            nn.ConvTranspose2d(16, im_enc_inp_sz, 3, 1, 0),  # Transposed Convolutional layer 3
        )

        self.apply(self._weights_init)

    def _weights_init(self, m):
        """
        Tensorflow/Keras-like initialization
        """
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xi, xv, state=None):
        """
        Forward pass of the SARNN module.
        Predicts the image, joint angle, and attention at the next time based on the image and joint angle at time t.
        Predict the image, joint angles, and attention points for the next state (t+1) based on
        the image and joint angles of the current state (t).
        By inputting the predicted joint angles as control commands for the robot,
        it is possible to generate sequential motion based on sensor information.

        Arguments:
            xi (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            xv (torch.Tensor): Input vector tensor of shape (batch_size, input_dim).
            state (tuple, optional): Initial hidden state and cell state of the LSTM cell.

        Returns:
            y_image (torch.Tensor): Decoded image tensor of shape (batch_size, channels, height, width).
            y_joint (torch.Tensor): Decoded joint prediction tensor of shape (batch_size, joint_dim).
            enc_pts (torch.Tensor): Encoded points tensor of shape (batch_size, k_dim * 2).
            dec_pts (torch.Tensor): Decoded points tensor of shape (batch_size, k_dim * 2).
            rnn_hid (tuple): Tuple containing the hidden state and cell state of the LSTM cell.
        """

        # Encode input image
        im_hid = self.im_encoder(xi)
        enc_pts, _ = self.pos_encoder(xi)

        # Reshape encoded points and concatenate with input vector
        enc_pts = enc_pts.reshape(-1, self.k_dim * 2)
        hid = torch.cat([enc_pts, xv], -1)

        rnn_hid = self.rec(hid, state)  # LSTM forward pass
        y_joint = self.decoder_joint(rnn_hid[0])  # Decode joint prediction
        dec_pts = self.decoder_point(rnn_hid[0])  # Decode points

        # Reshape decoded points
        dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
        heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
        hid = torch.mul(heatmap, im_hid)  # Multiply heatmap with image feature `im_hid`
        img_logits = decoder_image(hid)# Decode image
        if self.pos_as_mask:
           img_logits = img_logits[..., :-1]
           mask_logits = img_logits[..., -1]
 
        y_image = self.activation(img_logits)  
        return y_image, y_joint, enc_pts, dec_pts, rnn_hid
