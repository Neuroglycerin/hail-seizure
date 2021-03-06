% ICA-based estimation of LiNGAM model from data
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn*(nChn-1)/2) vector covariance of
%                             each pair of channels
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_ilingam(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('LiNGAM feature does not need any parameters. Dont provide any.');
end

% Initialise seed ---------------------------------------------------------
seedrng();

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Estimate the interaction weights
[B, stde, ci, k, Wout] = estimate(Dat.data);

% Extract the lower triangle (excluding diagonal: no self-causality)
nPairs = nChn*(nChn-1)/2;

% Initialise holding variable
featV = cell(3,1);
featV{1} = nan(1,nPairs);
featV{2} = nan(1,nChn);
featV{3} = nan(1,nChn);

% Add the causal order of the channels [ch5, ch2, ch3, ch1, ch4, ...]
featV{2}(1,:) = k;
% Add the order index of each channel [4th, 2nd, 3rd, 1st, 5th, ...]
[~, featV{3}(1,:)] = sort(k);

% Initialise pair counter
paircount = 0;
for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        % Insert into feature vector
        featV{1}(1,paircount) = B(iChn2,iChn1);
    end
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams.featnames = cell(3,1);
outparams.featnames{1} = 'connweights';
outparams.featnames{2} = 'causalorder';
outparams.featnames{3} = 'causalindex';

end