% Append feature to HDF5
% Append feature vector to HDF5 file of name and type
% Inputs : str subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str typ          - which datasegment type e.g. preictal, ictal, test
%        : str feature_name - name you wish to use for feature vector in HDF5
%        : vec featM        - feature_vector you want to add
% Outputs: void 
%

function [featM, f] = addToHDF5(subj, typ, feature_name, featM)

    settings = json.read('SETTINGS.json');
    subject_h5 = strcat(settings.TRAIN_DATA_PATH, '/', subj, settings.VERSION, '.h5');
    dataset = strcat('/', typ, '/', feature_name);
    h5create(subject_h5, dataset, ...
             size(featM), ...
             'Datatype', 'double');
    h5write(subject_h5, dataset, featM);
   
