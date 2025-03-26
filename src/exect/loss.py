import torch.nn as nn
import torch


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, n_mask: int = 0, *args, **kwargs):
        super().__init__()
        self.BCE = nn.BCEWithLogitsLoss(**kwargs)
        self.n_mask = n_mask

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            output (torch.Tensor): The output tensor from the model.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The masked loss tensor.
        """
        loss = self.BCE(output, target)
        return self._mask_loss(loss)

    def _mask_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Applies a mask to the loss tensor to set the loss to zero for the least relevant labels.

        Args:
            loss (torch.Tensor): The loss tensor.

        Returns:
            torch.Tensor: The masked loss tensor.
        """
        if self.n_mask == 0:
            return loss.mean()

        assert (
            self.n_mask >= 0
        ), f"self.n_mask is {self.n_mask} but must be greater than 0"

        bs = loss.shape[0]

        # sort loss row-wise and get indices
        loss_indices_sorted = torch.sort(loss, dim=1, descending=True)[1]

        # Set loss to zero for the least relevant labels
        loss[
            torch.arange(loss.size(0)).unsqueeze(1),
            loss_indices_sorted[:, self.n_mask :],
        ] = 0.0

        return loss.sum() / (bs * self.n_mask)



class MultiLabelCrossEntropyLoss(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.target = torch.tensor(0, dtype=torch.long).to(device)
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        ones_per_row = [torch.nonzero(row).squeeze().tolist() for row in target]
        zeros_per_row = [torch.nonzero(row==0.0).squeeze().tolist() for row in target]
        if weights is not None:
            weights = torch.tensor(weights)
        losses = []
        for i, (row, ones, zeros) in enumerate(zip(output, ones_per_row, zeros_per_row)):
            zero = row[zeros]
            if isinstance(ones, int):
                ones = [ones]
            for one in ones:
                if weights is not None:
                    # Create a new weight tensor for this instance
                    instance_weights = torch.cat([weights[one].unsqueeze(-1), weights[zeros]])
                    # Initialize CrossEntropyLoss with the instance-specific weight tensor
                    CE = nn.CrossEntropyLoss(weight=instance_weights).to(self.device)
                else:
                    CE = nn.CrossEntropyLoss().to(self.device)
                
                preds = torch.cat([row[one].unsqueeze(-1), zero])
                losses.append(CE(preds, self.target))

        return torch.stack(losses).mean()
 