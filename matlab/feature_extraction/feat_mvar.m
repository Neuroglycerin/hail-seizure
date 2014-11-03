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
    'maxlagdur' , [] );    % Duration of max lag for model (seconds)

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

if isempty(param.modelorder)
    if isempty(param.maxlagdur)
        % Default duration
        param.maxlagdur = 0.050; % 50ms
    end
    param.modelorder = round(param.maxlagdur * Dat.fs);
end
param.maxlagdur = param.modelorder / Dat.fs;

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Number of subfeatures
nSbf = 19;

% Do the MVAR fitting
[ARF, ~, PE] = mvar(Dat.data', param.modelorder, 2);

% Extract covariance matrix of innovation noise
C = PE(:,size(PE,2)+(1-nChn:0));

% Compute a bunch of frequency domain features based on MVAR model
[DC,DTF,PDC,GPDC,COH,PCOH,~,H,S,P,param.f] = fdMVAR(ARF, C, param.modelorder, Dat.fs);

% Initialise
sfsiz = [1, nChn, nChn, param.modelorder];
featV = nan([sfsiz nSbf]);
param.featnames = cell(1,1,1,1,nSbf);
iSbf = 0;
% Add features to where they belong
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(ARF,sfsiz);         param.featnames{iSbf} = 'ARF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(DC),sfsiz);     param.featnames{iSbf} = 'DC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(DC),sfsiz);   param.featnames{iSbf} = 'DCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(DTF),sfsiz);    param.featnames{iSbf} = 'DTF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(DTF),sfsiz);  param.featnames{iSbf} = 'DTFphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(PDC),sfsiz);    param.featnames{iSbf} = 'PDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(PDC),sfsiz);  param.featnames{iSbf} = 'PDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(GPDC),sfsiz);   param.featnames{iSbf} = 'GPDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(GPDC),sfsiz); param.featnames{iSbf} = 'GPDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(COH),sfsiz);    param.featnames{iSbf} = 'COH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(COH),sfsiz);  param.featnames{iSbf} = 'COHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(PCOH),sfsiz);   param.featnames{iSbf} = 'PCOH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(PCOH),sfsiz); param.featnames{iSbf} = 'PCOHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(H),sfsiz);      param.featnames{iSbf} = 'H';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(H),sfsiz);    param.featnames{iSbf} = 'Hphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(S),sfsiz);      param.featnames{iSbf} = 'S';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(S),sfsiz);    param.featnames{iSbf} = 'Sphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(P),sfsiz);      param.featnames{iSbf} = 'P';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(P),sfsiz);    param.featnames{iSbf} = 'Pphs';

if iSbf~=nSbf; error('Subfeature count mismatch'); end;

% ------------------------------------------------------------------------
% Set output parameters
outparams = param;

end