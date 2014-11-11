function preprocess_clean()

addpath(genpath('matlab'));

subjlst = subjnames();
% Loop over all subjects
for iSub=1:length(subjlst);
    main(subjlst{iSub});
end

end


function main(subj)

% Parameters --------------------------------------------------------------
order = 2; % filter order
dbgmde = 1; % debug mode

ictypes = {'preictal'; 'interictal'; 'test'};

% Get a list of all the filenames in all the ictypes ----------------------
fnamelist = {};
for iTyp=1:length(ictypes)
    icfnamelist = dircat(subj, ictypes{iTyp});
    fnamelist = [fnamelist(:); icfnamelist(:)];
end

% We need to always do the same cleaning to all files for the subject
% Check what cleaning is necessary for this subject
% Load the first file and see
Cntr = load(fnamelist{1});
f = fieldnames(Cntr);

if Cntr.(f{1}).sampling_frequency > 1000;
    dohighpass = true;
else
    dohighpass = false;
end
doremovelinenoise = testlinenoise(Cntr.(f{1}), fnamelist{1});

needcln = (dohighpass || doremovelinenoise);

if ~needcln
    fprintf('Subject %s does not need cleaning\n',subj);
end
if doremovelinenoise
    fprintf('Subject %s needs line noise removed\n',subj);
end
if dohighpass
    fprintf('Subject %s needs a highpass filter\n',subj);
end

writeclnmeta(subj,needcln,doremovelinenoise,dohighpass);

% Process each file -------------------------------------------------------

% For each datafile, load it up, do the separating matrix transformation,
% and save as a different file
nFle = length(fnamelist);
for iFle=1:nFle
    if dbgmde && mod(iFle,100)==0; fprintf('Processing file %3d/%3d\n',iFle,nFle); end
    % Load the saved matfile
    Cntr = load(fnamelist{iFle});
    % Matfile contains a structure named the same as the file
    f = fieldnames(Cntr);
    % Transform the data in the structure in the structure
    if doremovelinenoise
        Cntr.(f{1}) = removelinenoise(Cntr.(f{1}), order);
    end
    if dohighpass
        Cntr.(f{1}) = highpassfilter(Cntr.(f{1}), order);
    end
    % Label to show how the data was cleaned
    Cntr.(f{1}).iscln = true;
    Cntr.(f{1}).doremovelinenoise = doremovelinenoise;
    Cntr.(f{1}).dohighpass = dohighpass;
    % Get new file name
    clnfname = raw2clnfname(fnamelist{iFle});
    % Make directory if necessary
    if ~exist(fileparts(clnfname),'dir'); mkdir(fileparts(clnfname)); end;
    disp(clnfname);
    % Save cleaned copy
    save(clnfname,'-v7.3','-struct','Cntr');
end

end


function x = dircat(subj, ictyp)

[fnames, mydir] = subjtyp2dirs(subj, ictyp, 'raw');
nFle = length(fnames);
x = cell(nFle,1);
for iFle=1:nFle;
    x{iFle} = fullfile(mydir,fnames{iFle});
end

end


function Dat = removelinenoise(Dat, order)

Dat.data = butterfiltfilt(Dat.data', [55 65], Dat.sampling_frequency, order, 'stop', 'both')';

% Remove 180Hz line noise artifact harmonic
Dat.data = butterfiltfilt(Dat.data', [175 185], Dat.sampling_frequency, order, 'stop', 'both')';

% Remove 240Hz line noise artifact harmonic
Dat.data = butterfiltfilt(Dat.data', [235 245], Dat.sampling_frequency, order, 'stop', 'both')';

end


function Dat = highpassfilter(Dat, order)
    
Dat.data = butterfiltfilt(Dat.data', [0.1 Inf], Dat.sampling_frequency, order)';

end


function haslinenoise = testlinenoise(Dat, fnme)

% Parameters --------------------------------------------------------------
linefreq = 60; % frequency of line noise in Hz

% -------------------------------------------------------------------------
% CHANGE THIS TO CHECK PWELCH
haslinenoise = ~isempty(strfind(fnme,'Patient_1'));

end


function clnfname = raw2clnfname(fname)

% Decompose existing file
[pth,fname,ext] = fileparts(fname);

% Go up a directory, so we are before the subject name
[pth,subj] = fileparts(pth);

% Add cln folder between data directory and subject name
pth = fullfile(pth,'cln',subj);

% Add cln label in front of file name
fname = ['cln_' fname];

% Recombine new path, filename and extension
clnfname = fullfile(pth,[fname ext]);

end


function writeclnmeta(subj,needcln,doremovelinenoise,dohighpass)

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% -------------------------------------------------------------------------
% Load the settings file
settings = json.read(settingsfname);

% Work out the file name
mydir = fullfile(getRepoDir(), settings.MODEL_PATH);
Wfname = ['cleaningmetadata_' subj];

Wfnamefull = fullfile(mydir,Wfname);
Wfnamefull_log = fullfile(mydir,'log',[Wfname '_' datestr(now,30)]);

% Write the metadata
fprintf('Writing cleaning metadata to file\n  %s\n',Wfnamefull);
if ~exist(fileparts(Wfnamefull),'dir'); mkdir(fileparts(Wfnamefull)); end;
save(Wfnamefull,'needcln','doremovelinenoise','dohighpass');
if ~exist(fileparts(Wfnamefull_log),'dir'); mkdir(fileparts(Wfnamefull_log)); end;
save(Wfnamefull_log,'needcln','doremovelinenoise','dohighpass'); % Save a dated copy for posterity

end