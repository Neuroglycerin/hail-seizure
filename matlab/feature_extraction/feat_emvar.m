% Extended Multi-Variate Auto-Regressive model feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn X nChn X order X 38 vector
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
        param.maxlagdur = 0.030; % 30ms
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
nSbf = 2*19;

% Do the eMVAR fitting
% Method 2: Partial Correlation Estimation, Vieira-Morf using unbiased covariance estimates
[Bm,B0,Sw,Am,Su] = idMVAR0ng(Dat.data, param.modelorder, 2);

% Compute a bunch of frequency domain features based on MVAR model
[DC,DTF,PDC,GPDC,COH,PCOH,~,H,S,P,param.f] = fdMVAR(Am, Su, param.modelorder, Dat.fs);

% DC= Directed Coherence (Eq. 11)
% DTF= Directed Transfer Function (Eq. 11 but with sigma_i=sigma_j for each i,j)
% PDC= Partial Directed Coherence (Eq. 15 but with sigma_i=sigma_j for each i,j)
% GPDC= Generalized Partial Directed Coherence (Eq. 15)
% COH= Coherence (Eq. 3)
% PCOH= Partial Coherence (Eq. 3)
% H= Tranfer Function Matrix (Eq. 6)
% S= Spectral Matrix (Eq. 7)
% P= Inverse Spectral Matrix (Eq. 7)
% f= frequency vector

% Initialise
sfsiz = [1, nChn, nChn, param.modelorder];
featV = nan([sfsiz nSbf]);
param.featnames = cell(1,1,1,1,nSbf);

iSbf = 0;
% Add features to where they belong
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(Am,sfsiz);           param.featnames{iSbf} = 'ARF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(DC).^2,sfsiz);   param.featnames{iSbf} = 'DC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(DC),sfsiz);    param.featnames{iSbf} = 'DCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(DTF).^2,sfsiz);  param.featnames{iSbf} = 'DTF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(DTF),sfsiz);   param.featnames{iSbf} = 'DTFphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(PDC).^2,sfsiz);  param.featnames{iSbf} = 'PDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(PDC),sfsiz);   param.featnames{iSbf} = 'PDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(GPDC).^2,sfsiz); param.featnames{iSbf} = 'GPDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(GPDC),sfsiz);  param.featnames{iSbf} = 'GPDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(COH).^2,sfsiz);  param.featnames{iSbf} = 'COH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(COH),sfsiz);   param.featnames{iSbf} = 'COHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(PCOH).^2,sfsiz); param.featnames{iSbf} = 'PCOH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(PCOH),sfsiz);  param.featnames{iSbf} = 'PCOHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(H),sfsiz);       param.featnames{iSbf} = 'H';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(H),sfsiz);     param.featnames{iSbf} = 'Hphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(S),sfsiz);       param.featnames{iSbf} = 'S';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(S),sfsiz);     param.featnames{iSbf} = 'Sphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(P),sfsiz);       param.featnames{iSbf} = 'P';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(P),sfsiz);     param.featnames{iSbf} = 'Pphs';


% Compute a bunch of frequency domain features based on eMVAR model
[eDC,eDDC,ePDC,eDPDC,eCOH,ePCOH,eG,eS,eP,param.f] = fdMVAR0(Bm, B0, Sw, param.modelorder, Dat.fs);

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

% Add features to where they belong
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(Bm,sfsiz);            param.featnames{iSbf} = 'eARF';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eDC).^2,sfsiz);   param.featnames{iSbf} = 'eDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eDC),sfsiz);    param.featnames{iSbf} = 'eDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eDDC).^2,sfsiz);  param.featnames{iSbf} = 'eDDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eDDC),sfsiz);   param.featnames{iSbf} = 'eDDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(ePDC).^2,sfsiz);  param.featnames{iSbf} = 'ePDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(ePDC),sfsiz);   param.featnames{iSbf} = 'ePDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eDPDC).^2,sfsiz); param.featnames{iSbf} = 'eDPDC';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eDPDC),sfsiz);  param.featnames{iSbf} = 'eDPDCphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eCOH).^2,sfsiz);  param.featnames{iSbf} = 'eCOH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eCOH),sfsiz);   param.featnames{iSbf} = 'eCOHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(ePCOH).^2,sfsiz); param.featnames{iSbf} = 'ePCOH';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(ePCOH),sfsiz);  param.featnames{iSbf} = 'ePCOHphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eG),sfsiz);       param.featnames{iSbf} = 'eG';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eG),sfsiz);     param.featnames{iSbf} = 'eGphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eS),sfsiz);       param.featnames{iSbf} = 'eS';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eS),sfsiz);     param.featnames{iSbf} = 'eSphs';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(abs(eP),sfsiz);       param.featnames{iSbf} = 'eP';
iSbf=iSbf+1; featV(1,:,:,:,iSbf) = reshape(angle(eP),sfsiz);     param.featnames{iSbf} = 'ePphs';

if iSbf~=nSbf; error('Subfeature count mismatch'); end;

% ------------------------------------------------------------------------
% Set output parameters
outparams = param;

end