
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

    def yappend(self, new_y, new_x):
        """ Add a new y-value

        Args:
            new_y:

        Returns:

        """
        self.x.append(new_x)
        self.y.append(new_y)

        if not self.updated_since_plot:
            self.updated_since_plot = True

    def default(self, o):
        return o.__dict__

    def accumulate(self, sum, weight):
        self.current_sum += sum
        self.current_weight += weight

        if not self.accumlator_active:
            self.accumlator_active = True

    def reset_accumlator(self, new_x):
        if self.accumlator_active:

            self.y.append(self.current_sum / self.current_weight)
            self.x.append(new_x)
            self.current_weight = 0
            self.current_sum = 0
            self.accumlator_active = False
            self.updated_since_plot = True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.__dict__)

primitive = (int, str, bool, float)
def is_primitive(thing):
    return isinstance(thing, primitive)

class AutoStat(Stat):
    def __init__(self, x_counter, x_weight, x_plot, x_title="", y_title="", name="", plot=True, ymax=None, train=True):
        """ AutoStat - same as Stat, but don't need to specify the x-coords every time
            Specify the x_value once, which should be an ADDRESS of some object (not an actual number)

        Args:
            (x: the x plot values)
            x_counter: A TrainingCounter object
            x_weight (str): The attribute in the counter object that will be used for determining the weight
            x_title:
            y_title:
            name:
            plot:
            ymax:
            train: if the stat is a training one (the weighting is different; weighting is constant for test sets)
        """
        super().__init__(y=[None], x=[0], x_title=x_title, y_title=y_title, name=name, plot=plot, ymax=ymax)
        self.last_weight_step = 0
        self.x_counter = x_counter
        self.x_weight = x_weight
        self.x_plot = x_plot
        self.train = train

    def get_weight(self):
        if self.train:
            new_step = self.x_counter.__dict__[self.x_weight]
            weight = (new_step - self.last_weight_step)
            self.last_weight_step = new_step
        else:
            weight = self.x_weight
        if weight == 0:
            print("Error with weight - should be non-zero - using 1")
            weight = 1
        return weight

    def get_x(self):
        return self.x_counter.__dict__[self.x_plot]

    def accumulate(self, sum, weight=None):
        self.current_sum += sum

        if not self.accumlator_active:
            self.accumlator_active = True

    def reset_accumlator(self, new_x=None):
        if self.accumlator_active:
            weight = self.get_weight()

            # Update plot values
            self.y.append(self.current_sum / weight)
            self.x.append(self.get_x())

            # Reset Accumulator
            self.current_sum = 0
            self.accumlator_active = False
            self.updated_since_plot = True


class TrainingCounter:
    def __init__(self, instances_per_epoch=1, epochs=0, updates=0, instances=0):
        self.epoch = epochs
        self.updates = updates
        self.instances = instances
        self.instances_per_epoch = instances_per_epoch
        self.epoch_decimal =  self.instances/self.instances_per_epoch

    def update(self, epochs=0, instances=0, updates=0):
        self.epochs += epochs
        self.instances += instances
        self.updates += updates
        self.epoch_decimal = self.instances / self.instances_per_epoch


if __name__=='__main__':
    AutoStat()
    object.__dict__