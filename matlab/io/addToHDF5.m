% Append feature to HDF5
% Append feature vector to HDF5 file of name and type
% Script has error handling and allows overwriting of pre-existing datasets
% as long as datatype and size hasn't changed (HDF5 doesn't seem to support 
% changing this so we may have an issue with this at some point)
% Inputs : str subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str typ          - which datasegment type e.g. preictal, ictal, test
%        : str feat_name    - feature_name e.g. 'feat_cov'
%        : vec featM        - feature_vector you want to add
% Outputs: void 
%

function addToHDF5(subj, typ, feat_name, featM)

    settings = json.read('SETTINGS.json');
    subject_h5 = strcat(settings.TRAIN_DATA_PATH, '/', subj, settings.VERSION, '.h5');
    dataset = strcat('/', typ, '/', feat_name);

    % h5create will throw an error if the dataset already exists
    try
        h5create(subject_h5, dataset, ...
             size(featM), ...
             'Datatype', class(featM));    
        h5write(subject_h5, dataset, featM);

    catch 
        % if the dataset does exist and the try throws an exception then we
        % can just call h5write by itself to overwrite the dataset
        try
            h5write(subject_h5, dataset, featM);
        catch 
            % Unfortunately if the dataset has changed type or size this
            % will fail
            error('datasize or type has changed');
        end
    end
end
