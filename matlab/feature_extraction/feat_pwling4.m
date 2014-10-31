% Pairwise linear non-Gaussian causality modelling #4
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn*(nChn-1)/2) vector pairwise
%                             causality for each pair of channels
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_pwling4(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('Pairwise linear non-gaussian model feature does not need any parameters. Dont provide any.');
end
% Empty params output
outparams = struct([]);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Compute the pairwise linear non-gausian model with method #5
C = pwling(Dat.data,-4);

% Extract the upper triangle (excluding diagonal: no self-causality)
nPairs = nChn*(nChn-1)/2;

% Initialise holding variable
featV = nan(1,nPairs);

% Initialise pair counter
paircount = 0;
for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        % Insert into feature vector
        featV(1,paircount) = C(iChn1,iChn2);
    end
end

end