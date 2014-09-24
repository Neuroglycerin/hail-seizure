% Lists names of all files for given subject name and data ictal type
% in ascending numeric order
function [fnames, mydir, segIDs] = subjtyp2dirs(subj, typ)

% INPUT HANDLING ----------------------------------------------------------
if ~ismember(subj,subjnames())
    error('Bad subject name: %s',subj);
end
typ = ictyp2ictyp(typ);

% MAIN --------------------------------------------------------------------
% Create a directory system for the files required
fname_format = [subj '_' typ '_*.mat'];
mydir = fullfile(getDataDir, subj);

% Check directory exists
if ~exist(mydir,'dir')
    error('Non-existent directory: %s', mydir);
end

% Use dir function to list all files with this format in the directory
myfiles = dir(fullfile(mydir,fname_format));

% Go through the list of files in the directory and extract the names
% Initialise output variables
nFile = numel(myfiles);
fnames = cell(nFile,1);
segIDs = nan(nFile,1);
pat = '\w+_(\d+).mat'; % Regex format to identify the number on this file
for iFile = 1:nFile
    % Pick the file name out from the structure
    fnames{iFile} = myfiles(iFile).name;
    % Use regex to find the fileID number within the filename
    ret = regexp(myfiles(iFile).name, pat, 'tokens');
    % Convert the string (in the cell in the cell) into numeric type
    segIDs(iFile) = str2double(ret{1}{1});
end

% Sort segIDs so the filenames are in ascending numerical order
[segIDs,idx] = sort(segIDs,'ascend');
fnames = fnames(idx);

end