import os
from visdom import Visdom
import numpy as np
import torch
import warnings
import json


## Some functions stolen from https://github.com/theevann/visdom-save

class Plot(object):
    def __init__(self, title="", env_name="", config=None, port=8080):
        self.viz = Visdom(port=port, env=env_name)
        #self.viz.close()
        self.windows = {}
        self.title = title
        self.config = config
        self.env_name = env_name

    def register_plot(self, name, xlabel, ylabel, plot_type="line", ymax=None):
        self.windows[name] = {"xlabel":xlabel, "ylabel":ylabel, "title":name, "plot_type":plot_type}
        self.windows[name]["opts"] = dict(title=name, markersize=5, xlabel=xlabel, ylabel=ylabel)

        if ymax is not None:
            self.windows[name]["opts"]["layoutopts"] = dict(plotly=dict(yaxis=dict(range=[0, ymax])))


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
                opts=plot_d["opts"]
            )
            plot_d["plot"] = win
            self.windows["name"] = plot_d

    # LOADING
    def load_log(self, path):
        self.viz.replay_log(path)

    def load_all_env(self, root, keyword="visdom"):
        for d, ss, fs in os.walk(root):
            for f in fs:
                full_env = os.path.join(d, f)
                # Don't load "BSF" graphs, just complete graphs
                if full_env[-5:]==".json" and keyword in full_env and f != "losses.json" and "BSF_" not in full_env:
                    print("Loading {}".format(full_env))
                    self.viz.replay_log(full_env) # viz.load load the environment to viz

    def save_env(self, file_path=None, current_env=None, new_env=None):
        if file_path is None:
            file_path = os.path.join(self.config["results_dir"], "visdom.json")
        if current_env is None:
            current_env = self.env_name
            
        new_env = current_env if new_env is None else new_env
        #self.viz = Visdom(env=current_env) # get current env
        data = json.loads(self.viz.get_window_data())
        if len(data) == 0:
            print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
            return
    
        file = open(file_path, 'w+')
        for datapoint in data.values():
            output = {
                'win': datapoint['id'],
                'eid': new_env,
                'opts': {}
            }
    
            if datapoint['type'] != "plot":
                output['data'] = [{'content': datapoint['content'], 'type': datapoint['type']}]
                if datapoint['height'] is not None:
                    output['opts']['height'] = datapoint['height']
                if datapoint['width'] is not None:
                    output['opts']['width'] = datapoint['width']
            else:
                output['data'] = datapoint['content']["data"]
                output['layout'] = datapoint['content']["layout"]
    
            to_write = json.dumps(["events", output])
            file.write(to_write + '\n')
        file.close()

hwr_loss_title = "HWR Loss"
writer_loss_title = "Writer Loss"
total_loss_title = "Total Loss"

def initialize_visdom(env_name, config):
    if not config["use_visdom"]:
        return
    config["visdom_manager"] = Plot("Loss", env_name=env_name, config=config)
    config["visdom_manager"].register_plot(hwr_loss_title, "Instances", "Loss")
    config["visdom_manager"].register_plot(writer_loss_title, "Instances", "Loss")
    config["visdom_manager"].register_plot(total_loss_title, "Instances", "Loss")
    config["visdom_manager"].register_plot("Test Error Rate", "Epoch", "Loss", ymax=.2)
    return config["visdom_manager"]

def plot_loss(config, epoch, _hwr_loss, _writer_loss=None, _total_loss=None):
    visdom_manager = config["visdom_manager"]
    if not config["use_visdom"]:
        return

    # Plot writer recognizer loss and total loss
    if _writer_loss is None:
        _writer_loss = 0
    if _total_loss is None:
        _total_loss = _hwr_loss

    # Plot regular losses
    visdom_manager.update_plot(hwr_loss_title, [epoch], _hwr_loss)
    visdom_manager.update_plot(writer_loss_title, [epoch], _writer_loss)
    visdom_manager.update_plot(total_loss_title, [epoch], _total_loss)


if __name__=="__main__":
    plot = Plot("Test")
    plot.register_scatterplot("Loss", "Epoch", "Loss")

    for i in range(0,10):
        plot.update_plot("Loss", i, 2)