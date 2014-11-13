% Granger Causality feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn X nChn X nLag
%                             containing estimated G-causality from channel
%                             in dim3 to channel in dim2. The diagonal
%                             elements are the log-error given the past of
%                             this channel alone.
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_gcaus(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'modelorder', [] , ... % Model order
    'maxlagdur' , [] );    % Duration of max lag for model (seconds)

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

if isempty(param.modelorder)
    if isempty(param.maxlagdur)
        % Default duration
        param.maxlagdur = 0.030; % 30ms
    end
    param.modelorder = round(param.maxlagdur * Dat.fs);
end
param.maxlagdur = param.modelorder / Dat.fs;

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

E1 = nan(nChn, 1, param.modelorder+1);
% First parse all the single channels
for iChn1=1:nChn
    % Fit an autoregressive model for this channel
    [~, ~, PE] = mvar(Dat.data(iChn1,:)', param.modelorder, 2);
    % Find the error for this variable given its own history
    E1(iChn1,1,:) = PE(1,:);
end

E2 = ones(nChn, nChn, param.modelorder+1);
% Now parse all the pairs of channels
for iChn1=1:nChn
    % Assign the diagonal to be the square of the error for this channel
    % That way when we divide through by it later, we have the error again
    E2(iChn1,iChn1,:) = E1(iChn1,1,:).^2;
    % Only need to do each pair of channels once
    for iChn2=iChn1+1:nChn
        % Fit an autoregressive model for the pair of channels
        [~, ~, PE] = mvar(Dat.data([iChn1 iChn2],:)', param.modelorder, 2);
        % Find the error for the first channel when both are known
        E2(iChn1,iChn2,:) = PE(1,1:2:size(PE,2));
        % Find the error for the second channel when both are known
        E2(iChn2,iChn1,:) = PE(2,2:2:size(PE,2));
    end
end

% Initialise output
featV = nan(1,nChn,nChn,param.modelorder+1);
% Now compute the Granger Causality
featV(1,:,:,:) = log(bsxfun(@rdivide, E1, E2));

% ------------------------------------------------------------------------
% Set output parameters
outparams = param;

end