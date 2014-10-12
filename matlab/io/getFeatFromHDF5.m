% Get feature from HDF5
% Inputs : str  featname     - feature_name e.g. 'feat_cov'
%        : str  subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str  ictyp        - which datasegment type e.g. preictal, ictal, test
%        : str  modtyp       - which preprocessing model was used e.g. raw, ica
%        
% Outputs: array featM       - matrix of features, concatenated along first dim

function featM = getFeatFromHDF5(featname, subj, ictyp, modtyp)

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% Main --------------------------------------------------------------------
% Work out h5 filename
settings = json.read(settingsfname);
h5fnme = [modtyp '_' featname '_' settings.VERSION '.h5'];
h5fnme = fullfile(getRepoDir(), settings.TRAIN_DATA_PATH, h5fnme);

if ~exist(h5fnme,'file');
    error('HDF5 file does not exist');
end

% Get metadata structure
Info = h5info(h5fnme);

% Need to get a list of all the datasets
[fnames, jSub, jIct] = scrapeH5datasets(Info, subj, ictyp);
nSeg = numel(fnames);

% Check the size of the feature vector
featVsiz = Info.Groups(jSub).Groups(jIct).Datasets(1).Dataspace.Size;
% Make a feature matrix to house all the feature vectors
featMsiz = featVsiz;
featMsiz(1) = featMsiz(1)*nSeg;
featM = nan(featMsiz);

% Make a cell to use for assigning the vectors into the matrix
allcell = cell(1,length(featVsiz));
for iDim=1:length(featVsiz)
    allcell{iDim} = 1:featVsiz(iDim);
end

% Load all the data from the HDF5
for iSeg=1:nSeg
    lft = allcell;
    lft{1} = featVsiz(1)*(iSeg-1) + (1:featVsiz(1));
    featM(lft{:}) = h5read(h5fnme, ['/' subj '/' ictyp '/' fnames{iSeg}]);
end

end

% Get a list of all the datasets within the HDF5 for a given subject and
% ictal type
function [fnames, jSub, jIct] = scrapeH5datasets(Info, subj, ictyp)

% Pool through first level of groups to find which is our subject
nSub = numel(Info.Groups);
jSub = [];
for iSub=1:nSub
    if strcmp(['/' subj], Info.Groups(iSub).Name)
        jSub = iSub;
        break;
    end
end
if isempty(jSub)
    error('Subject %s is not in the HDF5 file',subj);
end

% Pool through second level of groups to find which is our ictal type
nIct = numel(Info.Groups(jSub).Groups);
jIct = [];
for iIct=1:nIct
    if strcmp(['/' subj '/' ictyp], Info.Groups(jSub).Groups(iIct).Name)
        jIct = iIct;
        break;
    end
end
if isempty(jIct)
    error('Ictal type %s is not in the HDF5 file',ictyp);
end

% Pool through names of all the datasets and list them
nSeg = numel(Info.Groups(jSub).Groups(jIct).Datasets);
fnames = cell(nSeg,1);
for iSeg=1:nSeg
    fnames{iSeg} = Info.Groups(jSub).Groups(jIct).Datasets(iSeg).Name;
end

end