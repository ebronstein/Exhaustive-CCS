import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy(fit_result, acc_name_prefix="", save_path=None):
    eval_histories = fit_result["eval_histories"]
    data = []

    for trial, history in enumerate(eval_histories):
        epochs = history["epoch"]
        for split in ["train", "test"]:
            for acc_dataset in ["sup_acc", "unsup_acc"]:
                acc_name = f"{acc_name_prefix}{split}_{acc_dataset}"
                acc_list = history[acc_name]
                for epoch, acc_val in zip(epochs, acc_list):
                    data.append(
                        {
                            "trial": trial,
                            "split": split,
                            "type": acc_dataset,
                            "epoch": epoch,
                            "accuracy": acc_val,
                        }
                    )
    df = pd.DataFrame(data)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    for ax, acc_dataset in zip(axs, ["sup_acc", "unsup_acc"]):
        sns.lineplot(
            data=df[df.type == acc_dataset],
            x="epoch",
            y="accuracy",
            style="split",
            ci="sd",
            markers=False,
            dashes=True,
            ax=ax,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(acc_dataset)

    fig.suptitle("Accuracy on Train and Test Splits")
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

    return df


def plot_history(
    fit_result, vars_to_plot=None, one_fig=False, save_path=None, logger=None
):
    if not fit_result["train_histories"] and logger is not None:
        logger.info("No train_histories found in fit_result")

    if vars_to_plot is None:
        vars_to_plot = fit_result["train_histories"][0].keys()
    else:
        if not all(
            var_name in fit_result["train_histories"][0] for var_name in vars_to_plot
        ):
            raise ValueError(
                f"vars_to_plot contains unknown variable names: {vars_to_plot}"
            )

    # Prepare the data for plotting
    data = []
    for var_name in vars_to_plot:
        for split, histories in zip(
            ["train", "test"],
            [fit_result["train_histories"], fit_result["eval_histories"]],
        ):
            for trial, history in enumerate(histories):
                var_history = history.get(var_name, [])
                if "epoch" in history:
                    epochs = history["epoch"]
                else:
                    epochs = range(len(var_history))
                for epoch, value in zip(epochs, var_history):
                    data.append(
                        {
                            "Epoch": epoch,
                            "Value": value,
                            "Type": var_name,
                            "Trial": trial,
                            "Split": split,
                        }
                    )

    df = pd.DataFrame(data)

    # Plot
    nrows = 1 if one_fig else len(vars_to_plot)
    fig, axs = plt.subplots(nrows, 1, figsize=(8, nrows * 6))

    if one_fig:
        sns.lineplot(
            data=df,
            x="Epoch",
            y="Value",
            hue="Type",
            style="Split",
            ci="sd",
            markers=False,
            dashes=True,
            ax=axs,
        )
        axs.set_title("History with Confidence Interval")
        axs.set_xlabel("Epoch")
        axs.legend(title="Value")
    else:
        if nrows == 1:
            axs = [axs]

        for ax, var_name in zip(axs, vars_to_plot):
            sns.lineplot(
                data=df[df["Type"] == var_name],
                x="Epoch",
                y="Value",
                style="Split",
                ci="sd",
                markers=False,
                dashes=True,
                ax=ax,
            )
            ax.set_title(f"{var_name} History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(var_name)

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

    return df
