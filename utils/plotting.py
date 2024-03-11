import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy(fit_result, save_path=None):
    sup_acc_df = pd.DataFrame(
        fit_result["sup_accuracies"], columns=["accuracy"]
    )
    sup_acc_df["dataset"] = "supervised"

    unsup_acc_df = pd.DataFrame(
        fit_result["unsup_accuracies"], columns=["accuracy"]
    )
    unsup_acc_df["dataset"] = "unsupervised"

    acc_df = pd.concat([sup_acc_df, unsup_acc_df], ignore_index=True)

    sns.violinplot(x="dataset", y="accuracy", data=acc_df)
    plt.ylim(0, 1)
    plt.suptitle("Supervised vs. Unsupervised Train Set Accuracy")

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def plot_history(
    fit_result, vars_to_plot=None, one_fig=False, save_path=None, logger=None
):
    if not fit_result["histories"] and logger is not None:
        logger.info("No histories found in fit_result")

    if vars_to_plot is None:
        vars_to_plot = fit_result["histories"][0].keys()
    else:
        if not all(
            var_name in fit_result["histories"][0] for var_name in vars_to_plot
        ):
            raise ValueError(
                f"vars_to_plot contains unknown variable names: {vars_to_plot}"
            )

    # Prepare the data for plotting
    data = []
    for var_name in vars_to_plot:
        for trial, history in enumerate(fit_result["histories"]):
            var_history = history.get(var_name, [])
            for epoch, value in enumerate(var_history):
                data.append(
                    {
                        "Epoch": epoch,
                        "Value": value,
                        "Type": var_name,
                        "Trial": trial,
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
            style="Type",
            ci="sd",
            markers=False,
            dashes=False,
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
                ci="sd",
                markers=False,
                dashes=False,
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
