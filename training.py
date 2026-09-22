"""Reusable PyTorch training loop with scheduling and early stopping."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch


def train(
    parameters: Iterable[torch.Tensor],
    loss_fn: Callable[[], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    scheduler=None,
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
    print_every: int = 100,
) -> list[float]:
    """Optimize parameters using a scalar loss returned by ``loss_fn``.

    The callable contains all problem-specific computations. When early stopping
    is enabled, the best parameter values are restored before returning.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("parameters must contain at least one tensor")

    best_loss = float("inf")
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    patience_counter = 0
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn()
        loss_value = loss.detach().item()
        loss_history.append(loss_value)

        # The loss and saved parameters refer to the same pre-update state.
        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            patience_counter = 0
        else:
            patience_counter += 1

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(loss_value)

        if print_every > 0 and (epoch + 1) % print_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Loss: {loss_value:.6e}, LR: {current_lr:.2e}"
            )

        if (
            early_stopping_patience is not None
            and patience_counter >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best loss: {best_loss:.6e}"
            )
            break

    with torch.no_grad():
        for parameter, best_parameter in zip(parameters, best_parameters):
            parameter.copy_(best_parameter)

    return loss_history
