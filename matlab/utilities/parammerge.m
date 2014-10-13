% S = parammerge(struct1, struct2, flg)
% Merges structures together
% flg overwrite: S has only fields in struct1, those also appearing in
%                struct2 are in S as in struct2
% flg union:     S has fields as in struct2, but those only appearing in
%                struct1 are in S as in struct1

function param = parammerge(defaultparam, inputparam, flg)

if nargin<3
    flg = 'overwrite';
end

param = defaultparam;

switch flg
    case 'overwrite'
        fnms = intersect(fieldnames(param), fieldnames(inputparam));
    case 'union'
        fnms = fieldnames(inputparam);
    otherwise
        error('Bad option');
end

for iFld=1:length(fnms)
    param(1).(fnms{iFld}) = inputparam.(fnms{iFld});
end

end