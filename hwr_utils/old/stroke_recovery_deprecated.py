from hwr_utils.stroke_recovery import *


def create_functions_from_strokes_old(strokes, time_interval=None, scale_time_distance=True):
    """ Takes in a stroke dictionary for one image

        time_interval (float): duration of upstroke events; None=original duration
        Returns:
             x(t), y(t), stroke_up_down(t), start_times
                 Note that the last start time is the end of the last stroke
    """
    x_list = []
    y_list = []
    t_list = []
    stroke_down = []
    stroke_down_times = []

    t_offset = 0
    start_times = []
    epsilon = 1e-8

    # Epsilon is the amount of time before or after a stroke for the interpolation
    # Time between strokes must be greater than epsilon, or interpolated points will result
    if time_interval < epsilon:
        time_interval = epsilon * 3

    distance = 0

    # euclidean distance metric
    distance_metric = lambda x, y: ((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2) ** (1 / 2)

    # Loop through each stroke
    for i, stroke_dict in enumerate(strokes):
        xs = np.asarray(stroke_dict["x"])
        ys = np.asarray(stroke_dict["y"])
        distance += np.sum(distance_metric(xs, ys))

        x_list += stroke_dict["x"]
        y_list += stroke_dict["y"]

        # Set duration for "upstroke" events
        if not time_interval is None and i > 0:
            next_start_time = stroke_dict["time"][0]
            last_end_time = t_list[-1]
            t_offset = time_interval + last_end_time - next_start_time
            t_list_add = [t + t_offset for t in stroke_dict["time"]]
        else:
            t_list_add = stroke_dict["time"]

        t_list += t_list_add
        start_times += [t_list_add[0]]

        ## Stroke up/down times
        # Add a stroke up
        if time_interval > 2 * epsilon:
            stroke_down += [0] + [1] * len(stroke_dict["x"]) + [0]
            stroke_down_times += [t_list_add[0] - epsilon] + t_list_add + [t_list_add[-1] + epsilon]
        else:
            stroke_down += [1] * len(stroke_dict["x"])
            stroke_down_times += t_list_add

    # Add the last time to the start times
    start_times += [t_list_add[-1]]

    # Have interpolation not move after last point
    x_list += [x_list[-1]]
    y_list += [y_list[-1]]
    t_list += [t_list[-1] + 20]
    stroke_down += [0, 0]
    stroke_down_times += [stroke_down_times[-1] + epsilon, stroke_down_times[-1] + 20]

    stroke_down_func = interpolate.interp1d(stroke_down_times, stroke_down)
    x_func = interpolate.interp1d(t_list, x_list)
    y_func = interpolate.interp1d(t_list, y_list)
    return x_func, y_func, stroke_down_func, start_times


def read_stroke_xml_old(path, max_stroke_count=None):
    """
    Args:
        path: XML path to stroke file

    Returns:
        list of lists of dicts: each dict contains a stroke, keys: x,y, time
    """
    root = ET.parse(path).getroot()
    all_strokes = root[1]
    stroke_lists = []
    start_end_strokes_lists = []

    # If not constrained by number of strokes, set max_stroke_count to full desired_num_of_strokes window
    if max_stroke_count is None:
        max_stroke_count = len(all_strokes) - 1

    for i in range(len(all_strokes) - max_stroke_count):
        strokes = all_strokes[i:i + max_stroke_count]
        stroke_list = []
        min_time = float(strokes[0][0].attrib["time"])
        last_time = 0
        stroke_delay = 0  # time between strokes
        start_end_strokes = []  # list of start times and end times between strokes; one before sequence starts!

        for stroke in strokes:
            x_coords = []
            y_coords = []
            time_list = []

            for i, point in enumerate(stroke):
                # print("Points", len(strokes))
                x, y, time = point.attrib["x"], point.attrib["y"], point.attrib["time"]
                x_coords.append(int(x))
                y_coords.append(-int(y))

                if i == 0:  # no time passes between strokes!
                    min_time += float(time) - min_time - last_time - .001
                    start_end_strokes.append((last_time, float(time) - min_time))

                next_time = float(time) - min_time

                if time_list and next_time == time_list[-1]:
                    next_time += .001
                    assert next_time > time_list[-1]

                # No repeated times
                if time_list and next_time <= time_list[-1]:
                    next_time = time_list[-1] + .001

                time_list.append(next_time)
            last_time = time_list[-1]
            stroke_list.append({"x": x_coords, "y": y_coords, "time": time_list})

        stroke_lists.append(stroke_list)
        start_end_strokes_lists.append(start_end_strokes)

    return stroke_lists, start_end_strokes_lists

def convert_strokes(stroke_list):
    """ Convert the stroke dict to 3 lists

    Args:
        stroke_list (list): list of dicts, each dict contains a stroke, keys: x,y, time
    Returns:
        tuple of array-likes: x coordinates, y coordinates, times

    """
    x, y, time = [], [], []
    [x.extend(key["x"]) for key in stroke_list]
    [y.extend(key["y"]) for key in stroke_list]
    [time.extend(key["time"]) for key in stroke_list]
    return np.array(x), np.array(y), np.array(time)


def zero_center_data(my_array, _max=1):
    """ Max/min rescale to -1,1 range

    Args:
        my_array:

    Returns:

    """
    return ((my_array - np.min(my_array)) / array_range(my_array) - .5) * 2 * _max


def extract_gts(path, instances=50, max_stroke_count=None):
    """ Take in xml with strokes, output ordered target coordinates
        Parameterizes x & y coordinates as functions of t
        Any t can be selected; strokes are collapsed so there's minimal time between strokes

        Start stroke flag - true for first point in stroke
        End stroke flag - true for last point in stroke
        ** A single point can have both flags!

    Args:
        path (str): path to XML
        instances (int): number of desired coordinates

    Returns:
        x-array, y-array
    """
    stroke_lists, start_end_strokes_lists = read_stroke_xml(path, max_stroke_count=max_stroke_count)

    output_gts = []
    output_stroke_lists = []
    output_xs_to_ys = []

    for stroke_list, start_end_strokes in zip(stroke_lists, start_end_strokes_lists):
        x, y, time = convert_strokes(stroke_list)

        # find dead timezones
        # make x and y independently a function of t
        time_continuum = np.linspace(np.min(time), np.max(time), instances)
        x_func = interpolate.interp1d(time, x)
        y_func = interpolate.interp1d(time, y)
        begin_stroke = []
        start_end_strokes_backup = start_end_strokes.copy()

        # Each time a stop/start break is met, we've started a new stroke
        # We start with a stop/start break
        end_stroke_override = False

        strokes_left = len(start_end_strokes)
        for i, t in enumerate(time_continuum):
            for ii, (lower, upper) in enumerate(start_end_strokes):
                if t < lower:
                    break  # same stroke, go to next timestep
                elif t > lower and t < upper:
                    if abs(t - lower) < abs(t - upper):
                        t = lower
                    else:
                        t = upper
                    time_continuum[i] = t
                if t >= upper:  # only happens on last item of stroke
                    start_end_strokes = start_end_strokes[ii + 1:]
                    break

            # Don't use strokes that can't help anymore
            # print(len(start_end_strokes), len(start_end_strokes[ii:]), start_end_strokes)

            if strokes_left > len(start_end_strokes):
                strokes_left = len(start_end_strokes)
                begin_stroke.append(1)
            else:
                begin_stroke.append(0)

        end_stroke = begin_stroke.copy()[1:] + [1]
        begin_stroke = np.array(begin_stroke)
        end_stroke = np.array(end_stroke)
        end_of_sequence = np.zeros(time_continuum.shape[0])
        end_of_sequence[-1] = 1

        # print(end_of_sequence.shape, end_stroke.shape, begin_stroke.shape)
        # print(begin_stroke)
        # print(end_stroke)
        # print(end_of_sequence)

        x_range = array_range(x_func(time_continuum))
        y_range = array_range(y_func(time_continuum))

        assert len(zero_center_data(x_func(time_continuum))) == len(zero_center_data(y_func(time_continuum))) == len(
            begin_stroke) == len(end_stroke) == len(end_of_sequence)
        output = np.array(
            [zero_center_data(x_func(time_continuum)), zero_center_data(y_func(time_continuum)), begin_stroke,
             end_stroke, end_of_sequence])

        output_gts.append(output)
        output_stroke_lists.append(stroke_list)
        output_xs_to_ys.append(x_range / y_range)

    return output_gts, output_stroke_lists, output_xs_to_ys

def array_range(my_array):
    return np.max(my_array)-np.min(my_array)
