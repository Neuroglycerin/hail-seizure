% Cross-channel Correlation coefficient Eigenvalues of Fourier Transform
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x nChn vector of eigenvalues for correlation
%                             coefficient matrix on FFT of data
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_FFTcorrcoefeig(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('FFT correlation coefficient eigenvalue feature does not need any parameters. Dont provide any.');
end

% Main --------------------------------------------------------------------
% Take fast-fourier transform
Dat.data = fft(Dat.data,[],2);

% Pass to corrcoef function
[featV,outparams] = feat_corrcoefeig(Dat, inparams);

end