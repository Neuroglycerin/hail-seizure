% Cross-channel Correlation coefficient feature of Fourier Transform
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x (nChn*(nChn-1)/2) vector correlation
%                             coefficient of each non-trivial pair of channels
%                             after taking FFT
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_FFTcorrcoef(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('FFT Correlation coefficient feature does not need any parameters. Dont provide any.');
end

% Main --------------------------------------------------------------------
% Take fast-fourier transform
Dat.data = fft(Dat.data,[],2);

% Pass to corrcoef function
[featV,outparams] = feat_corrcoef(Dat, inparams);

end