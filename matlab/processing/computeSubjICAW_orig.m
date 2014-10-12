% Computes and saves the ICA weights for a subject
% Uses all datapoints in all datafiles for the subject
% Calls FastICA, but is very memory intensive
% Can be used on smaller subject datasets as a sanity check to make sure
% I didn't break the FastICA code when moving it here
function W = computeSubjICAW_orig(subj, dbgmde, do_icafiles)

% =========================================================================
% Initialise random number generator
rng(7);

% =========================================================================
% Default inputs ----------------------------------------------------------
if nargin<3;
    do_icafiles = false; % Whether to prevent overwritting files
end
if nargin<2;
    dbgmde   = true; % Whether to write progress to screen
end
pltmde = 'off'; % Whether to plot ICA progress

% Parameters --------------------------------------------------------------
ictypall = {'inter','pre','test'}; % Which ictypes exist
ictypuse = {'inter','pre','test'}; % Which ictypes to use for the ICA model

% Input Handling ----------------------------------------------------------
% Check subject is one for which there is data
if ~ismember(subj,subjnames())
    error('Bad subject name: %s',subj);
end

% Main --------------------------------------------------------------------

if dbgmde; fprintf('Performing ICA processing for subject %s\n',subj); end

% Find ICA model ----------------------------------------------------------
% Get a list of all the filenames in the ictypes we will use --------------
fnamelist = {};
for iTyp=1:length(ictypuse)
    icfnamelist = dircat(subj, ictypuse{iTyp});
    fnamelist = [fnamelist(:); icfnamelist(:)];
end

% Load all the data and put it into an enormous matrix --------------------
nFle = length(fnamelist);
fleSrtIdx = nan(1,nFle+1);
fleSrtIdx(1) = 1;
if dbgmde; fprintf('Loading %d raw data files\n',nFle); end
for iFle=1:nFle
    if dbgmde && (mod(iFle,50)==0 || iFle==1 || iFle==2);
        fprintf('Loading file %3d/%3d\n',iFle,nFle);
    end
    % Load the saved matfile
    Dat = load(fnamelist{iFle});
    % Matfile contains a structure named the same as the file
    f = fieldnames(Dat);
    % Check how many datapoints there are
    datlen = size(Dat.(f{1}).data,2);
    % Initialise the holding matrix
    if iFle==1
        if dbgmde; fprintf('Making RAM available for large dataset\n'); end
        mixedsig = nan(size(Dat.(f{1}).data,1), datlen*nFle);
    end
    % Take the data out of the structure in the structure
    mixedsig(:,fleSrtIdx(iFle)+(0:datlen-1)) = Dat.(f{1}).data;
    % Note where the next file should start its entry
    fleSrtIdx(iFle+1) = fleSrtIdx(iFle) + datlen;
end

% Cut off any excess length
if size(mixedsig,2)~=fleSrtIdx(end)-1
    if dbgmde; fprintf('Cropping dataset appropriately\n'); end
    % mixedsig = mixedsig(:,1:fleSrtIdx(end)-1);
    mixedsig(:,fleSrtIdx(end):end) = [];
end

% =========================================================================
% Perform ICA on the enormous dataset -------------------------------------
if dbgmde; fprintf('Running FastICA\n'); end
[~, A, W] = fastica(mixedsig, 'displayMode', pltmde);
clear mixedsig;
% =========================================================================

% Write the ICA matrix to file --------------------------------------------
[Wfname,Wfnamefull_log] = getWfname(subj);
if dbgmde; fprintf('Writing weight matrix to file\n  %s\n',Wfname); end
if ~exist(fileparts(Wfname),'dir'); mkdir(fileparts(Wfname)); end;
save(Wfname,'W','A'); % Overwrite the copy to be used in transformations
if ~exist(fileparts(Wfnamefull_log),'dir'); mkdir(fileparts(Wfnamefull_log)); end;
save(Wfnamefull_log,'W','A'); % Save a dated copy for posterity

% Quit if we are not writing any files
if ~do_icafiles; return; end;

% Process each file -------------------------------------------------------
% Get a list of all the filenames in all the ictypes ----------------------
fnamelist = {};
for iTyp=1:length(ictypall)
    icfnamelist = dircat(subj, ictypall{iTyp});
    fnamelist = [fnamelist(:); icfnamelist(:)];
end

% For each datafile, load it up, do the separating matrix transformation,
% and save as a different file
nFle = length(fnamelist);
for iFle=1:nFle
    if dbgmde && mod(iFle,100)==0; fprintf('Processing file %3d/%3d\n',iFle,nFle); end
    % Load the saved matfile
    Dat = load(fnamelist{iFle});
    % Matfile contains a structure named the same as the file
    f = fieldnames(Dat);
    % Transform the data in the structure in the structure
    Dat.(f{1}).data = W * Dat.(f{1}).data;
    % Get new file name
    icafname = raw2icafname(fnamelist{iFle});
    % Make directory if necessary
    if ~exist(fileparts(icafname),'dir'); mkdir(fileparts(icafname)); end;
    % Save ICA
    save(icafname,'-v7.3','-struct','Dat');
end

if dbgmde; fprintf('Finished ICA processing for subject %s\n',subj); end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = dircat(subj, ictyp)

[fnames, mydir] = subjtyp2dirs(subj, ictyp, 'raw');
nFle = length(fnames);
x = cell(nFle,1);
for iFle=1:nFle;
    x{iFle} = fullfile(mydir,fnames{iFle});
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wfnamefull,Wfnamefull_log] = getWfname(subj)

% Declarations
settingsfname = 'SETTINGS.json';

% Load the settings file
settings = json.read(settingsfname);

mydir = fullfile(getRepoDir(), settings.MODEL_PATH);
Wfname = ['ica_weights_' subj];

Wfnamefull = fullfile(mydir,Wfname);
Wfnamefull_log = fullfile(mydir,'log',[Wfname '_' datestr(now,30)]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function icafname = raw2icafname(fname)

% Decompose existing file
[pth,fname,ext] = fileparts(fname);

% Go up a directory, so we are before the subject name
[pth,subj] = fileparts(pth);

% Add ica folder between data directory and subject name
pth = fullfile(pth,'ica',subj);

% Add ica label in front of file name
fname = ['ica_' fname];

% Recombine new path, filename and extension
icafname = fullfile(pth,[fname ext]);

end
