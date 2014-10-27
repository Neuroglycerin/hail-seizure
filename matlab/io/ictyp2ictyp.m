% Convert from ictal type shorthand to cannonical ictal type string
% (Cannonical ictal type string is the one used in the filenames)
function typ = ictyp2ictyp(typ)

if ischar(typ)
    typ = lower(typ);
elseif ~isnumeric(typ) || (~isequal(typ,0) && ~isequal(typ,1))
    error('Ictal type shorthand should be string or 0 or 1');
end

switch typ
    case {1,'p','pre','preictal'}
        typ = 'preictal';
        
    case {'pp','pseudopre','pseudopreictal'}
        typ = 'pseudopreictal';
        
    case {0,'i','inter','interictal'}
        typ = 'interictal';
        
    case {'pi','pseudointer','pseudointerictal'}
        typ = 'pseudointerictal';
        
    case {'?','t','test'}
        typ = 'test';
        
    otherwise
        error('Bad type: %s',typ);
        
end

end