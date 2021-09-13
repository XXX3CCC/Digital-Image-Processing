import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class TrainParams:
    """
    :ivar optim_type: optimizer type: 0: SGD, 1: ADAM

    :ivar load_weights:
        0: train from scratch
        1: load and test
        2: load if it exists and continue training

    :ivar save_criterion:  when to save a new checkpoint
        0: max validation accuracy
        1: min validation loss
        2: max training accuracy
        3: min training loss

    :ivar lr: learning rate
    :ivar eps: term added to the denominator to improve numerical stability in ADAM optimizer

    :ivar valid_ratio: fraction of training data to use for validation
    :ivar valid_gap: no. of training epochs between validations

    :ivar vis: visualize the input and reconstructed images during validation and testing;
    only works for offline runs since colab doesn't support cv2.imshow
    """

    def __init__(self):
        self.batch_size = 128
        self.optim_type = 1
        self.lr = 1e-3
        self.momentum = 0.9
        self.n_epochs = 1000
        self.eps = 1e-8
        self.weight_decay = 0
        self.save_criterion = 0
        self.load_weights = 0
        self.valid_gap = 1
        self.valid_ratio = 0.2
        self.weights_path = './checkpoints/model.pt'
        self.vis = 0


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        """
        add your code here
        """
        pass

    def init_weights(self):
        """
        add your code here
        """
        pass

    def forward(self, x):
        """
        add your code here
        """
        pass
