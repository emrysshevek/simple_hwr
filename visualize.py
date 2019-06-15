from visdom import Visdom
import numpy as np
import torch
import warnings

class Plot(object):
    def __init__(self, title, port=8080):
        self.viz = Visdom(port=port)
        #self.viz.close()
        self.windows = {}
        self.title = title

    def register_plot(self, name, xlabel, ylabel, plot_type="line"):
        self.windows[name] = {"xlabel":xlabel, "ylabel":ylabel, "title":name, "plot_type":plot_type}

    def update_plot(self, name, x, y):

        # Create plot if not registered
        try:
            plot_d = self.windows[name]
        except:
            warnings.warn("Plot not found, creating new plot")
            plot_d = {"xlabel":"X", "ylabel":"Y", "plot_type":"scatter"}

        plotter = self.viz.scatter if plot_d["plot_type"] == "scatter" else self.viz.line
        data = {"X": np.asarray(x), "Y": np.asarray([y])} if plot_d["plot_type"] == "line" else {"X": np.asarray([x, y])}

        ## Update plot
        if "plot" in plot_d.keys():
            plotter(
                **data,
                win=plot_d["plot"],
                update="append"
            )
        else: # Create new plot
            win = plotter(
                **data,
                opts=dict(title=name, markersize=5, xlabel=plot_d["xlabel"], ylabel=plot_d["ylabel"])
            )
            plot_d["plot"] = win
            self.windows["name"] = plot_d

if __name__=="__main__":
    plot = Plot("Test")
    plot.register_scatterplot("Loss", "Epoch", "Loss")

    for i in range(0,10):
        plot.update_plot("Loss", i, 2)
