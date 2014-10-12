% Append feature to HDF5
% Append feature vector to HDF5 file of name and type
% Script has error handling and allows overwriting of pre-existing datasets
% as long as datatype and size hasn't changed (HDF5 doesn't seem to support
% changing this so we may have an issue with this at some point)
% Inputs : cell featM        - feature_vector you want to add
%        : str  subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str  ictyp        - which datasegment type e.g. preictal, ictal, test
%        : str  featname    - feature_name e.g. 'feat_cov'
%        : str  modtyp       - which preprocessing model was used
%        
% Outputs: void

function addToHDF5(featM, subj, ictyp, featname, modtyp, inparams)
    
    % Declarations ------------------------------------------------------------
    settingsfname = 'SETTINGS.json';
    
    % Default inputs ----------------------------------------------------------
    if nargin<6
        inparams = [];
    end
    
    % Input handling ----------------------------------------------------------
    if ~isempty(inparams); error('Cant handle input parameters'); end;
    
    % Cannonicalise ictal type
    ictyp = ictyp2ictyp(ictyp);
    
    % Main --------------------------------------------------------------------
    settings = json.read(settingsfname);
    % Use the version ID currently in the settings file
    h5fnme = getFeatH5fname(featname, modtyp, settings.VERSION);
    
    nFle = size(featM, 1);
    
    for i=1:nFle
        seg_name = featM{i, 1};
        dataset = strcat('/', subj, '/', ictyp, '/', seg_name);
        data = featM{i, 2};
        
        h5writePlus(h5fnme, dataset, data);
        
    end
    
end
