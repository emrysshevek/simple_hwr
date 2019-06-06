from utils import pickle_it, unpickle_it
import os
import json
from xml.dom import minidom
from collections import defaultdict
import pickle

path = r"./data/prepare_IAM_Lines/xml"

def get_writers(xml_obj):
    return xml_obj.getElementsByTagName('form')[0].getAttribute("writer-id")

def loop_through_xmls(path, process_func=get_writers):
    output_dict = defaultdict(list)

    for d,s,fs in os.walk(path):
        for fn in fs:

            curr_path = os.path.join(d, fn)
            xml_obj = minidom.parse(curr_path)
            output_dict[process_func(xml_obj)].append(fn[:-4])

    return output_dict
    #with open(curr_path, "r") as f:
    #my_func


if __name__=="__main__":
    output = loop_through_xmls(path)
    pickle_it(output, "./data/prepare_IAM_Lines/writer_IDs.pickle")
    print(output)
