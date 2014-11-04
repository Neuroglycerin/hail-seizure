#!/usr/bin/env python

import python.utils as utils #contains mainly parsers
import json
import glob
import train
import csv

def main(models=None):
    #load the settings
    settings = json.load(open('SETTINGS.json', 'r'))

    if models == None:
        #get the paths of available models:
        models = glob.glob("model/*")
        # try saying this six times fast
        models = [model for model in models if 'model_' in model]
        # split each to match expected format
        models = [model.split("/")[-1] for model in models]

    subjects = settings['SUBJECTS']
    features = settings['FEATURES']
    #load the data
    data = utils.get_data(features,settings)
    #iterate over subjects
    predictiondict = {}
    for subject in subjects:
        #get the right model name (probably):
        subjectmodel = [model for model in models if subject in model][0]
        #load the trained model:
        model = utils.read_trained_model(subjectmodel)
        #build test set
        X,segments = utils.build_test(subject,features,data)
        #make predictions
        predictions = model.predict_proba(X)
        for segment,prediction in zip(segments,predictions):
            predictiondict[segment] = prediction

    #write the results to the submission csv
    with open("output/submission.csv", "w") as f:
        c = csv.writer(f)
        c.writerow(['clip','preictal'])
        for segment in predictiondict.keys():
            c.writerow([segment,"%s"%predictiondict[segment][-1]])

if __name__=='__main__':
    main()
