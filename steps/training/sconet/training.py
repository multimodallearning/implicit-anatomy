import torch

from steps.training.training import BaseTrainer

from util.nn.pytorch.losses import CELoss

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network models
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device

    '''

    def __init__(self, model, optimizer, device=None,loss=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = CELoss() if loss is None else loss

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        eval_dict = self.compute_loss(data)
        eval_dict['loss'].backward()
        self.optimizer.step()

        eval_dict['loss'] = eval_dict['loss'].item()
        return eval_dict

    @torch.no_grad()
    def eval_step(self, data):
        """Performs an evaluation step."""
        self.model.eval()

        device = self.device
        eval_dict = {}

        body_points = data["body_points"].to(device)
        body_offsets = data["body_offsets"].to(device)
        query_points = data["query_points"].to(device)
        query_labels = data["query_labels"].to(device)

        logits = self.model(
            query_points,
            body_points,
            body_offsets,
        )

        loss = self.loss(logits, query_labels).mean()

        eval_dict["loss"] = loss.item()

        # TODO: compute metric

        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss and metric during training.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        eval_dict = {}

        body_points = data["body_points"].to(device)
        body_offsets = data["body_offsets"].to(device)
        query_points = data["query_points"].to(device)
        query_labels = data["query_labels"].to(device)

        logits = self.model(
            query_points,
            body_points,
            body_offsets,
        )

        loss = self.loss(logits, query_labels).mean()

        eval_dict["loss"] = loss

        return eval_dict
