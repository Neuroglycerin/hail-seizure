% Pairwise linear non-Gaussian causality modelling #1
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn*(nChn-1)/2) vector pairwise
%                             causality for each pair of channels
%        :[struct outparams]- structure with fields listing parameters used
%
% NB: Excludes the variance terms. Use feat_var for single channel
% variance.

function [featV,outparams] = feat_pwling1(Dat, inparams)

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

% Take Z-score of each variable
Dat.data = bsxfun(@minus, Dat.data, mean(Dat.data,2));
Dat.data = bsxfun(@rdivide, Dat.data, std(Dat.data,[],2)+eps);

%Compute covariance matrix
C = cov(Dat.data');

% Extract the upper triangle (excluding diagonal: no self-causality)
nPairs = nChn*(nChn-1)/2;

% Initialise holding variable
featV = nan(1,nPairs);

% Parse maxentropy of each singleton
ME = nan(nChn,1);
for iChn=1:nChn
    ME(iChn) = mentappr(Dat.data(iChn,:));
end

% Initialise pair counter
paircount = 0;
for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        % Get this thing which relates the two channels in each direction
        res1 = (Dat.data(iChn2,:)-C(iChn2,iChn1)*Dat.data(iChn1,:));
        res2 = (Dat.data(iChn1,:)-C(iChn1,iChn2)*Dat.data(iChn2,:));
        % Compute difference in maxentropy of each direction
        LR = ME(iChn2)-ME(iChn1)-mentappr(res1)+mentappr(res2);
        % Insert into feature vector
        featV(1,paircount) = LR;
    end
end

end