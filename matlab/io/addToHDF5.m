% Append feature to HDF5
% Append feature vector to HDF5 file of name and type
% Script has error handling and allows overwriting of pre-existing datasets
% as long as datatype and size hasn't changed (HDF5 doesn't seem to support
% changing this so we may have an issue with this at some point)
% Inputs : cell featM        - feature_vector you want to add
%        : str  subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str  ictyp        - which datasegment type e.g. preictal, ictal, test
%        : str  feat_name    - feature_name e.g. 'feat_cov'
%        : str  modtyp       - which preprocessing model was used
%        
% Outputs: void
%

function addToHDF5(featM, subj, ictyp, feat_name, modtyp, inparams)

    % Default inputs ----------------------------------------------------------
    if nargin<6
        inparams = [];
    end

    % Input handling ----------------------------------------------------------
    if ~isempty(inparams); error('Cant handle input parameters'); end;

    % Declarations ------------------------------------------------------------
    settingsfname = 'SETTINGS.json';

    % Main --------------------------------------------------------------------
    settings = json.read(settingsfname);
    h5fnme = fullfile(getRepoDir(), settings.TRAIN_DATA_PATH, [subj, settings.VERSION, '.h5']);

    nFle = size(featM, 1);

    for i=1:nFle
        seg_name = featM{i, 1};
        dataset = strcat('/', ictyp, '/', modtyp, '_', feat_name, '/', seg_name);
        data = featM{i, 2};
        % h5create will throw an error if the dataset already exists
        try
            h5create(h5fnme, dataset, ...
                size(data), ...
                'Datatype', class(data));
            h5write(h5fnme, dataset, data);
        catch ME
            % if the dataset does exist and the try throws an exception then we
            % can just call h5write by itself to overwrite the dataset
            try
                h5write(h5fnme, dataset, featM);
            catch ME
                % Unfortunately if the dataset has changed type or size this
                % will fail
                error('datasize or type has changed');
            end
        end
    end
end
