% Extended Multi-Variate Auto-Regressive model feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn X nChn X order X 19 vector
%                             containing eMVAR and frequency components
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_emvar(Dat, inparams)

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
        param.maxlagdur = 0.030; % 50ms
    end
    param.modelorder = round(param.maxlagdur * Dat.fs);
end
param.maxlagdur = param.modelorder / Dat.fs;

% Initialise seed ---------------------------------------------------------
seedrng();

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Number of subfeatures
nSbf = 19;

% Do the MVAR fitting
[eBm,eB0,eSw] = idMVAR0ng(Dat.data, param.modelorder);

% Compute a bunch of frequency domain features based on eMVAR model
[EDC,DDC,EPDC,DPDC,COH,PCOH,G,S,P,param.f] = fdMVAR0(eBm, eB0, eSw, param.modelorder, Dat.fs);


% EDC= Extended Directed Coherence (Eq. 26)
% DDC= Delayed Directed Coherence (Eq. 29)
% EPDC= Extended Partial Directed Coherence (Eq. 27)
% DPDC= Delayed Partial Directed Coherence (Eq. 29)
% COH= Coherence (Eq. 3)
% PCOH= Partial Coherence (Eq. 3)
% G= Transfer Function Matrix (after Eq. 22)
% S= Spectral Matrix (Eq. 23)
% P= Inverse Spectral Matrix (Eq. 23)
% f= frequency vector

% Initialise
sfsiz = [1, nChn, nChn, param.modelorder];
featV = nan([sfsiz nSbf]);
param.featnames = cell(1,1,1,1,nSbf);
iSbf = 0;
% Add features to where they belong
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(eBm,sfsiz);          param.featnames{iSbf} = 'ARF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(EDC).^2,sfsiz);  param.featnames{iSbf} = 'EDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(EDC),sfsiz);   param.featnames{iSbf} = 'EDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(DDC).^2,sfsiz);  param.featnames{iSbf} = 'DDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(DDC),sfsiz);   param.featnames{iSbf} = 'DDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(EPDC).^2,sfsiz); param.featnames{iSbf} = 'EPDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(EPDC),sfsiz);  param.featnames{iSbf} = 'EPDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(DPDC).^2,sfsiz); param.featnames{iSbf} = 'DPDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(DPDC),sfsiz);  param.featnames{iSbf} = 'DPDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(COH).^2,sfsiz);  param.featnames{iSbf} = 'COH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(COH),sfsiz);   param.featnames{iSbf} = 'COHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(PCOH).^2,sfsiz); param.featnames{iSbf} = 'PCOH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(PCOH),sfsiz);  param.featnames{iSbf} = 'PCOHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(G),sfsiz);       param.featnames{iSbf} = 'G';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(G),sfsiz);     param.featnames{iSbf} = 'Gphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(S),sfsiz);       param.featnames{iSbf} = 'S';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(S),sfsiz);     param.featnames{iSbf} = 'Sphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(P),sfsiz);       param.featnames{iSbf} = 'P';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(P),sfsiz);     param.featnames{iSbf} = 'Pphs';

if iSbf~=nSbf; error('Subfeature count mismatch'); end;

% ------------------------------------------------------------------------
% Set output parameters
outparams = param;

end