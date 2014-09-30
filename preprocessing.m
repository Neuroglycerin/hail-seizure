%% Matlab script to pre-process raw data and output serialized features

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

% Parse settings.json
settings = json.read('SETTINGS.json');

subjects = subjnames();
types = {'preictal'; 'interictal'; 'test'};
feature_funcs = {@feat_corrcoef; ...
                 @feat_cov; ...
                 @feat_psd};
            
% for each subject create and h5 file in train
% within each h5 file create 3 groups for each type
% within each group put a dataset from each func
for i = 1:numel(subjects)
    subject_h5 = strcat(settings.TRAIN_DATA_PATH, '/', subjects{i}, settings.VERSION, '.h5');
    for j = 1:numel(types)
         parfor k = 1:numel(feature_funcs)
            featM = feature_funcs{k}(subjects{i}, types{j});
            dataset = strcat('/', types{j}, '/', func2str(feature_funcs{k}));
            h5create(subject_h5, dataset, ...
                                    size(featM), ...
                                    'Datatype', 'double');
            h5write(subject_h5, dataset, featM);
         end
     end
end
