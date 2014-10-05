% func should be a handle to a feature function which accepts two inputs
% the datastructure, Dat, and a structure of input parameters

function [featM, outparams] = computeFeatM(featfunc, subj, ictyp, modtyp, inparams)

% Input handling ----------------------------------------------------------
if nargin<5
    inparams = struct([]); % Empty struct
end
if ischar(featfunc)
    featfunc = str2func(featfunc);
end

% Setup -------------------------------------------------------------------
% Get the preprocessing function to use
ppfunc = getPreprocFunc(modtyp, subj);

% Get a list of files
[fnames, mydir, segIDs] = subjtyp2dirs(subj, ictyp);
nFle = length(fnames);

% Process the first segment so we know what the output size should be
% Load this segment
% Dat = loadSegFile(fullfile(mydir,fnames{1}));
% Apply the preprocessing model
% Dat = ppfunc(Dat);
% Compute the feature
% [featV,outparams] = featfunc(Dat, inparams);

% Initialise the holding variable
% featM = nan(nFle,numel(featV));
% Add the feature for the first segment file
% featM(1,:) = featV;

% Parallelised feature computation ----------------------------------------
% Loop over each segment, computing feature and adding to matrix

featM = cell(nFle, 2);

for iFle=1:nFle
    % Load this segment
    Dat = loadSegFile(fullfile(mydir,fnames{iFle}));
    % Apply the preprocessing model
    Dat = ppfunc(Dat);
    % Compute the feature
    feat_vec = featfunc(Dat, inparams);
    featM(iFle, :) = {fnames{iFle}, feat_vec};
end

end