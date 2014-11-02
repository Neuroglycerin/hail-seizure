function func = getPreprocFunc(modtyp, subj)

% Input handling ----------------------------------------------------------
if ~ischar(modtyp);
    error('Modelling type should be a character string');
end

% Main --------------------------------------------------------------------
switch lower(modtyp)
    case 'raw'
        func = @(x)x;
    case 'ica'
        func = @(x) raw2ica(x, subj);
    case 'csp'
        func = @(x) raw2csp(x, subj);
    otherwise
        error('Unfamiliar model preprocessing: %s',modtyp);
end

end