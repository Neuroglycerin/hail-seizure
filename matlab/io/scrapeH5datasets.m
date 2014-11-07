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
    error('getFeatFromHDF5:missingSubject','Subject %s is not in the HDF5 file',subj);
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
    error('getFeatFromHDF5:missingIctal','Ictal type %s is not in the HDF5 file',ictyp);
end

% Stop if we only need to know the data is available, not what the fnames are
if nargout==0; return; end;

% Pool through names of all the datasets and list them
nSeg = numel(Info.Groups(jSub).Groups(jIct).Datasets);
fnames = cell(nSeg,1);
for iSeg=1:nSeg
    fnames{iSeg} = Info.Groups(jSub).Groups(jIct).Datasets(iSeg).Name;
end

end