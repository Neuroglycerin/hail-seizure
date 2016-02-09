#!/usr/bin/env python3

import optparse
import numpy as np
import csv
import json
import os


def main(settings, output):
    """
    Function to take a json with a
    list of csvs and average them to
    produce a merged set of predictions.
    Crude bagging.
    Input:
    * settings - json settings file
    * output - output filename (will always be in "output" directory)
    Output:
    * merged - dictionary of results
    """
    # load json
    with open(settings) as f:
        outputcsvs = json.load(f)

    # dictionary of names filled with
    # dictionaries of predictions
    outputs = {}
    for fname in outputcsvs:
        with open(fname) as f:
            c = csv.reader(f)
            fl = next(c)
            outputs[fname] = {}
            for l in c:
                outputs[fname][l[0]] = float(l[1])
    # then open up the output csv and write the merged
    # result, while storing it in the merged dictionary
    merged = {}
    segments = list(outputs[outputcsvs[0]].keys())
    with open(os.path.join("output", output), "w") as cf:
        c = csv.writer(cf)
        c.writerow(fl)
        for s in segments:
            many = []
            for fname in outputcsvs:
                many.append(outputs[fname][s])
            mnval = np.mean(many)
            c.writerow([s, mnval])
            merged[s] = mnval

    return merged

if __name__ == "__main__":
    # parse cmdline args
    parser = optparse.OptionParser()
    # settings file
    parser.add_option("-s", "--settings",
                      action="store",
                      dest="settings",
                      default="SETTINGS.json",
                      help="Settings file to use in JSON format (default="
                            "SETTINGS.json)")
    # output file
    parser.add_option("-o", "--output",
                      action="store",
                      dest="output",
                      default="merged.csv",
                      help="Output file to write csv to (default="
                            "merged.csv)")
    # get the opts and the args
    (opts, args) = parser.parse_args()

    # call the main function with these two options
    main(opts.settings, opts.output)
