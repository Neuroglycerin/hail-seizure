function func = getPreprocFunc(modtyp, subj)

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
        func = @(x) raw2ica(x, subj);
    case 'csp'
        func = @(x) raw2csp(x, subj);
    case 'cln'
        func = @(x) raw2cln(x);
    case 'dwn'
        func = @(x) raw2dwn(x);
    otherwise
        error('Unfamiliar model preprocessing: %s',submodtyp);
end

if ~isempty(K)
    func2 = getPreprocFunc(modtyp(K(1)+1:end), subj);
    func = @(x) func2(func(x));
end

end