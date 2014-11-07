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
        func = @(x)x;
    case 'ica'
        func = @(x) raw2ica(x, subj, fullmodlst);
    case 'csp'
        func = @(x) raw2csp(x, subj, fullmodlst);
    case 'cln'
        func = @(x) raw2cln(x);
    case 'dwn'
        func = @(x) raw2dwn(x);
    otherwise
        error('Unfamiliar model preprocessing: %s',submodtyp);
end

if ~isempty(K)
    func2 = getPreprocFunc(modtyp(K(1)+1:end), subj, fullmodlst);
    func = @(x) func2(func(x));
end

end