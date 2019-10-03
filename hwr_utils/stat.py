class Stat:
    def __init__(self, y, x, x_title="", y_title="", name="", plot=True, ymax=None, accumulator_freq=None):
        """

        Args:
            y (list): iterable (e.g. list) for storing y-axis values of statistic
            x (list): iterable (e.g. list) for storing x-axis values of statistic (e.g. epochs)
            x_title:
            y_title:
            name (str):
            plot (str):
            ymax (float):
            accumulator_freq: when should the variable be accumulated (e.g. each epoch, every "step", every X steps, etc.


        """
        super().__init__()
        self.y = y
        self.x = x
        self.current_weight = 0
        self.current_sum = 0
        self.accumlator_active = False
        self.updated_since_plot = False
        self.accumulator_freq = None # epoch or instances; when should this statistic accumulate?

        # Plot details
        self.x_title = x_title
        self.y_title = y_title
        self.ymax = ymax
        self.name = name
        self.plot = plot
        self.plot_update_length = 1 # add last X items from y-list to plot

    def yappend(self, new_item):
        self.y.append(new_item)
        if not self.updated_since_plot:
            self.updated_since_plot = True

    def default(self, o):
        return o.__dict__

    def accumulate(self, sum, weight, step=None):
        self.current_sum += sum
        self.current_weight += weight

        if not self.accumlator_active:
            self.accumlator_active = True

        if step:
            self.x.append(step)

    def reset_accumlator(self):
        if self.accumlator_active:
            # print(self.current_weight)
            # print(self.current_sum)
            self.y += [self.current_sum / self.current_weight]
            self.current_weight = 0
            self.current_sum = 0
            self.accumlator_active = False
            self.updated_since_plot = True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.__dict__)
