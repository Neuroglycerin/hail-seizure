function func = getPreprocFunc(modtyp, subj, fullmodlst)

% If this is the first call, the full list of preprocessing is the same is
% the current list
if nargin<3
    fullmodlst = modtyp;
end

% Input handling ----------------------------------------------------------
if ~ischar(modtyp);
    error('Modelling type should be a character string');
end

% Main --------------------------------------------------------------------
% Commas separate different preprocessing functions to use
K = strfind(modtyp,',');
if ~isempty(K)
    submodtyp = modtyp(1:K(1)-1);
else
    submodtyp = modtyp;
end

switch lower(submodtyp)
    case 'raw'
        % Raw. Leave as is.
        func = @(x)x;
    case 'cln'
        % Clean. Remove line noise and really low frequencies.
        func = @(x) raw2cln(x, subj);
    case 'dwn'
        % Downsample to 400Hz
        func = @(x) raw2dwn(x);
    case 'ica'
        % Independent Component Analysis
        func = @(x) raw2ica(x, subj, fullmodlst);
    case 'icadr'
        % ICA with sources reduced to 8
        func = @(x) raw2dr(raw2ica(x, subj, fullmodlst));
    case 'csp'
        % Common Spatial Patterns
        func = @(x) raw2csp(x, subj, fullmodlst);
    case 'cspdr'
        % CSP with channels reduced to 8
        func = @(x) raw2dr(raw2csp(x, subj, fullmodlst));
    otherwise
        error('Unfamiliar model preprocessing: %s',submodtyp);
end

if ~isempty(K)
    func2 = getPreprocFunc(modtyp(K(1)+1:end), subj, fullmodlst);
    func = @(x) func2(func(x));
end

end

function Dat = raw2dr(Dat)

% Isolate the first 8 channels and discard the rest
Dat.data = Dat.data(1:8,:);

end