% Correlation coefficient of band envelope amplitude 
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             nBnd*2 cell
%                             odd: 1 X (nChn-1)*nChn/2 matrix, upper-triangle of corrcoef
%                             even: 1 X nChn matrix, eigenvalues of corrcoef
%        :[struct outparams]- structure with fields listing parameters used
%
% NB: Excludes the variance terms. Use feat_var for single channel
% variance.

function [featV,outparams] = feat_ampcorrcoef(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------

% Bands used in Howbert et al., 2014
defaultbands = struct(...
    'delta',      [ 1.0    4.0],  ...
    'theta',      [ 4.0    8.0],  ...
    'alpha',      [ 8.0   12.0],  ...
    'beta',       [12.0   30.0],  ...
    'low_gamma',  [30.0   70.0],  ...
    'high_gamma', [70.0  180.0]);

defparams = struct(...
    'order'  , 2   ,... % Filter order
    'bands'  , defaultbands);

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Check what bands we have
bandnames = fieldnames(param.bands);
nBnd = length(bandnames);

% Fill use the upper triangle (excluding diagonal)
nPairs = nChn*(nChn-1)/2;

% Initialise holding variable
featV = cell(2*nBnd,1);
param.featnames = cell(2*nBnd,1);

% Loop over each band
for iBnd=1:nBnd
    % Filter within this band
    fltDat = butterfiltfilt(Dat.data', param.bands.(bandnames{iBnd}), Dat.fs, param.order)';
    % Take phase with Hilbert transform
    fltDat = abs(hilbert(fltDat'))';
    % Compute the normalised covariance
    C = corrcoef(fltDat');
    % Initialise output
    featV{iBnd*2-1} = nan(1,nPairs);
    featV{iBnd*2  } = nan(1,nChn);
    param.featnames{iBnd*2-1} = bandnames{iBnd};
    % Initialise pair counter
    paircount = 0;
    % Extract the upper triangle
    for iChn1=1:nChn
        for iChn2=(iChn1+1):nChn
            % Track how many pairs we have done
            paircount = paircount + 1;
            % Insert into feature vector
            featV{iBnd*2-1}(1,paircount) = C(iChn1,iChn2);
        end
    end
    % Compute eigenvalues
    featV{iBnd*2}(1,:) = sort(eig(C));
    param.featnames{iBnd*2} = [bandnames{iBnd} '-eig'];
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;

end