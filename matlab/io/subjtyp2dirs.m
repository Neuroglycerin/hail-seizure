% Lists names of all files for given subject name and data ictal type
% in ascending numeric order
function [fnames, mydir, segIDs] = subjtyp2dirs(subj, ictyp, modtyp)

% Parameters --------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% DEFAULT INPUTS ----------------------------------------------------------
if nargin<3 || isempty(modtyp)
    modtyp = 'raw';
end

% INPUT HANDLING ----------------------------------------------------------
% Check subject is one for which there is data
if ~ismember(subj,subjnames())
    error('Bad subject name: %s',subj);
end
% Convert shorthand ictyp to cannonical
ictyp = ictyp2ictyp(ictyp);

% Load the settings file
settings = json.read(settingsfname);

% MAIN --------------------------------------------------------------------
mydir = getDataDir();

% Load cleaning parameters
ClnMeta = loadClnMeta(subj);

if ~isempty(strfind(modtyp,'cln')) && ClnMeta.needcln && ...
        ~strcmp(settings.VERSION,'_v1') && ~strcmp(settings.VERSION,'_v2')
    % Pre-cleaned file
    mydir = fullfile(mydir,'cln');
    fname_format = ['cln_' subj '_' ictyp '_*.mat'];
else
    % Raw file
    fname_format = [subj '_' ictyp '_*.mat'];
end

% Each subject is in its own subfolder
mydir = fullfile(mydir, subj);

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