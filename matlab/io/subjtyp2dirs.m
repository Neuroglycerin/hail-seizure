% Lists all files (in dir format) for given subject name and data ictal type
function [myfiles,mydir] = subjtyp2dirs(subj,typ)

% INPUT HANDLING ----------------------------------------------------------
if ~ismember(subj,subjnames())
    error('Bad subject name');
end
switch typ
    case {1,'p','pre','preictal'}
        fulltyp = 'preictal';
    case {0,'i','inter','interictal'}
        fulltyp = 'interictal';
    case {'t','test'}
        fulltyp = 'test';
    otherwise
        error('Bad type');
end

% MAIN --------------------------------------------------------------------
% Create a directory system for the files required
fname_format = [subj '_' fulltyp '_*.mat'];
mydir = fullfile(getDataDir, subj);

% Check directory exists
if ~exist(mydir,'dir')
    error('Non-existent directory: %s',mydir);
end

% Use dir function to list all files with this format in the directory
myfiles = dir(fullfile(mydir,fname_format));

end