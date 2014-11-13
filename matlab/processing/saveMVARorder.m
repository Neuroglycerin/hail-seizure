function avgorder = saveMVARorder(subj, modtyp)

% Declarations
settingsfname = 'SETTINGS.json';

% Load the settings file
settings = json.read(settingsfname);

mydir = fullfile(getRepoDir(), settings.MODEL_PATH);

fname = ['mvarorder_' modtyp '_' subj];
fnamefull = fullfile(mydir,fname);
fnamefull_log = fullfile(mydir,'log',[fname '_' datestr(now,30)]);

[allsbc, allfpe, allavg, avgorder, avgorder1, avgorder0] = computeMVARorder(subj, modtyp);

if ~exist(fileparts(fnamefull),'dir'); mkdir(fileparts(fnamefull)); end;
save(fnamefull, 'allsbc', 'allfpe', 'allavg', 'avgorder', 'avgorder1', 'avgorder0');
if ~exist(fileparts(fnamefull_log),'dir'); mkdir(fileparts(fnamefull_log)); end;
save(fnamefull_log, 'allsbc', 'allfpe', 'allavg', 'avgorder', 'avgorder1', 'avgorder0');

end

function [allsbc, allfpe, allavg, avgorder, avgorder1, avgorder0] = computeMVARorder(subj, modtyp)

% Setup -------------------------------------------------------------------
% Get the preprocessing function to use
ppfunc = getPreprocFunc(modtyp, subj);

% Get a list of files
[fnames1       ] = subjtyp2dirs(subj, 'preictal', modtyp);
[fnames0, mydir] = subjtyp2dirs(subj, 'interictal', modtyp);

fnames = [fnames1(:); fnames0(:)];
nFle = length(fnames);

allsbc = nan(nFle,1);
allfpe = nan(nFle,1);
allavg = nan(nFle,1);

parfor iFle=1:nFle
    
    % Load this segment
    Dat = loadSegFile(fullfile(mydir,fnames{iFle}));
    % Apply the preprocessing model
    Dat = ppfunc(Dat);
    
    if Dat.fs<1000
        maxorder = 60;
    else
        maxorder = 60;
    end
    
    [~, ~, ~, sbc, fpe] = arfit2(Dat.data', 1, maxorder);
    
    [~,allsbc(iFle)] = min(sbc);
    [~,allfpe(iFle)] = min(fpe);
    [~,allavg(iFle)] = min(sbc.*fpe);
    
end

avgorder1 = median(allavg(1:length(fnames1)));
avgorder0 = median(allavg(length(fnames1)+1:end));
avgorder  = median(allavg);

end
