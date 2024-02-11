import numpy as np


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])

def adder(df, model, prefix, method, prompt_level, train, test, accuracy, std, ece, location, layer, loss, sim_loss, cons_loss):
    return df.append({
                "model": model,
                "prefix": prefix,
                "method": method,
                "prompt_level": prompt_level,
                "train": train,
                "test": test,
                "accuracy": accuracy,
                "std": std,
                "ece": ece,
                "location": location,
                "layer": layer,
                "loss": loss,
                "sim_loss": sim_loss,
                "cons_loss": cons_loss,
            }, ignore_index=True)
