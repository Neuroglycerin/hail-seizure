% Multi-Variate Auto-Regressive model feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn X nChn X order X 21 vector
%                             containing MVAR and frequency components
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_mvar(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'modelorder', [] , ... % Model order
    'maxlag'    , [] );    % Duration of max lag for model (seconds)

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

if isempty(param.modelorder)
    if ~isempty(param.maxlag)
        % Default duration
        param.maxlag = 0.050; % 50ms
    end
    param.modelorder = param.maxlag * Dat.fs;
end
param.maxlag = ceil(param.modelorder / Dat.fs);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Number of subfeatures
nSbf = 21;

% Do the MVAR fitting
[ARF, ~, PE] = mvar(Dat.data', modelorder, 2);

% Extract covariance matrix of innovation noise
C = PE(:,size(PE,2)+(1-nChn:0));

% Compute a bunch of frequency domain features based on MVAR model
[DC,DTF,PDC,GPDC,COH,PCOH,~,H,S,P,param.f] = fdMVAR(ARF, C, modelorder, Dat.fs);

% Initialise
featV = nan(1, nChn, nChn, modelorder, nSbf);
param.featnames = cell(1,1,1,1,nSbf);
iSbf = 0;
% Add features to where they belong
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = ARF;         param.featnames{iSbf} = 'ARF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(DC);     param.featnames{iSbf} = 'DC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(DC);   param.featnames{iSbf} = 'DCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(DTF);    param.featnames{iSbf} = 'DTF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(DTF);  param.featnames{iSbf} = 'DTFphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(PDC);    param.featnames{iSbf} = 'PDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(PDC);  param.featnames{iSbf} = 'PDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(GPDC);   param.featnames{iSbf} = 'GPDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(GPDC); param.featnames{iSbf} = 'GPDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(COH);    param.featnames{iSbf} = 'COH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(COH);  param.featnames{iSbf} = 'COHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(PCOH);   param.featnames{iSbf} = 'PCOH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(PCOH); param.featnames{iSbf} = 'PCOHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(H);      param.featnames{iSbf} = 'H';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(H);    param.featnames{iSbf} = 'Hphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(S);      param.featnames{iSbf} = 'S';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(S);    param.featnames{iSbf} = 'Sphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = abs(P);      param.featnames{iSbf} = 'P';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = angle(P);    param.featnames{iSbf} = 'Pphs';

if iSbf~=nSbf; error('Subfeature count mismatch'); end;

% ------------------------------------------------------------------------
% Set output parameters
outparams = param;

end