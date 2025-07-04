import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from typing import Callable


@eqx.filter_jit
def compute_accuracy(
    model, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(
    model,
    testloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    metric_fn: Callable,
):
    """Taken from https://docs.kidger.site/equinox/examples/mnist/

    Args:
        model: model to evaluate
        testloader (torch.utils.data.DataLoader): data loader for the test set
        loss_fn (Callable): loss function
        metric_fn (Callable): metric function

    Returns:
        tuple: average loss and average metric
    """
    avg_loss = 0
    avg_metric = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss_fn` and `metric_fn`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss_fn(model, x, y)
        avg_metric += metric_fn(model, x, y)
    return avg_loss / len(testloader), avg_metric / len(testloader)


def evaluate_accuracy(
    model,
    testloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    return evaluate(model, testloader, loss_fn, compute_accuracy)


def train(
    model,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    loss_fn: Callable,
    evaluate_fn: Callable,
):
    """Taken from https://docs.kidger.site/equinox/examples/mnist/

    Args:
        model: the model to train
        trainloader (torch.utils.data.DataLoader): data loader for the train set
        testloader (torch.utils.data.DataLoader): data loader for the test set
        optim (optax.GradientTransformation): optimizer
        steps (int): number of steps to train
        print_every (int): frequency of log
        loss_fn (Callable): loss function
        evaluate_fn (Callable): evaluate function

    Returns:
        the new trained model

    """
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x
        y = y
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate_fn(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model
