"""Model evaluation utilities."""

import torch


class Evaluator:
    """Evaluates model performance with standard metrics."""

    def __init__(self, criterion, device):
        self.criterion = criterion
        self.device = device

    def evaluate(self, model, dataloader):
        """Compute metrics: MSE, MAE, MAPE, SMAPE, RSE, CORR."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = model(X_batch)

                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view_as(y_batch)

                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)

                all_preds.append(y_pred.cpu())
                all_targets.append(y_batch.cpu())

        all_preds = torch.cat(all_preds, dim=0).squeeze()
        all_targets = torch.cat(all_targets, dim=0).squeeze()

        epsilon = 1e-8
        num_samples = len(all_targets)

        mse = total_loss / num_samples
        mae = torch.mean(torch.abs(all_targets - all_preds)).item()
        mape = (
            torch.mean(torch.abs((all_targets - all_preds) / (all_targets + epsilon)))
            * 100
        ) / num_samples
        smape_numerator = torch.abs(all_preds - all_targets)
        smape_denominator = torch.abs(all_targets) + torch.abs(all_preds) + epsilon
        smape = torch.mean(2 * smape_numerator / smape_denominator) * 100
        rse_numerator = torch.sum((all_preds - all_targets) ** 2)
        rse_denominator = (
            torch.sum((all_targets - torch.mean(all_targets)) ** 2) + epsilon
        )
        rse = (rse_numerator / rse_denominator).item()
        vx = all_targets - torch.mean(all_targets)
        vy = all_preds - torch.mean(all_preds)
        corr = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + epsilon
        )

        return {
            "MSE": mse,
            "MAE": mae,
            "MAPE": mape.item(),
            "SMAPE": smape.item(),
            "RSE": rse,
            "CORR": corr.item(),
        }
