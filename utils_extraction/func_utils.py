import numpy as np


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])


def train_adder(
    df,
    model,
    prefix,
    method,
    prompt_level,
    train,
    test,
    location,
    layer,
    loss,
    sim_loss,
    cons_loss,
):
    """Add a row to the dataframe.

    Args:
        df: The dataframe to add a row to.
        model: The model name.
        prefix: The prefix.
        method: The method used.
        prompt_level: The prompt level.
        train: The training set.
        test: The test set.
        accuracy: The accuracy.
        std: The standard deviation of the accuracy.
        ece: The expected calibration error.
        ece_flip: The expected calibration error if the probabilities are flipped
            (i.e. 1 - p instead of p).
        location: The location in the prompt from which the hidden states are used.
        layer: The layer.
        loss: The loss.
        sim_loss: The similarity loss.
        cons_loss: The consistency loss.
    """
    return df.append(
        {
            "model": model,
            "prefix": prefix,
            "method": method,
            "prompt_level": prompt_level,
            "train": train,
            "test": test,
            "location": location,
            "layer": layer,
            "loss": loss,
            "sim_loss": sim_loss,
            "cons_loss": cons_loss,
        },
        ignore_index=True,
    )


def eval_adder(
    df,
    model,
    prefix,
    method,
    prompt_level,
    train,
    test,
    accuracy,
    std,
    ece,
    ece_flip,
    location,
    layer,
):
    """Add a row to the dataframe.

    Args:
        df: The dataframe to add a row to.
        model: The model name.
        prefix: The prefix.
        method: The method used.
        prompt_level: The prompt level.
        train: The training set.
        test: The test set.
        accuracy: The accuracy.
        std: The standard deviation of the accuracy.
        ece: The expected calibration error.
        ece_flip: The expected calibration error if the probabilities are flipped
            (i.e. 1 - p instead of p).
        location: The location in the prompt from which the hidden states are used.
        layer: The layer.
        loss: The loss.
        sim_loss: The similarity loss.
        cons_loss: The consistency loss.
    """
    return df.append(
        {
            "model": model,
            "prefix": prefix,
            "method": method,
            "prompt_level": prompt_level,
            "train": train,
            "test": test,
            "accuracy": accuracy,
            "std": std,
            "ece": ece,
            "ece_flip": ece_flip,
            "location": location,
            "layer": layer,
        },
        ignore_index=True,
    )
