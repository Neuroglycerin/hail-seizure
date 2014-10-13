%MAXEL  Largest element.
%   For vectors and matrices, MAXEL(X) is the largest element in X.
% 
%   [Y,I] = MAXEL(X), or equivalently [Y,I] = MAXEL(X,'ind'), returns the index
%   of the maximum element in X. If the matrix contains more than one 
%   maximal element, the index of the first one is returned.
%
%   [Y,S] = MAXEL(X,'sub') returns an array of the subscripts of the maximum
%   element in X. If the matrix contains more than one maximal element, 
%   the subscripts of the first one are returned.
%
%   [Y,C] = MAXEL(X,'subc') returns a cell array of the subscripts of the 
%   maximum element in X. If the matrix contains more than one maximal element, 
%   the subscripts of the first one are returned.
% 
%   When X is complex, the maximum is computed using the magnitude
%   MAXEL(ABS(X)). In the case of equal magnitude elements, then the phase
%   angle MAXEL(ANGLE(X)) is used.
% 
%   NaN's are ignored when computing the maximum. When all elements in X
%   are NaN's, then the first one is returned as the maximum.
% 
%   Example: If X = [2 8 4   then maxel(X) is 9,
%                    7 3 9]
% 
%       [Y,I] = maxel(X) gives Y = 9 and I = 6.
%
%       [Y,C] = maxel(X,'subc') gives Y = 9 and C = {2 3}.
%       We then find X(C{:}) is 9.
% 
%   See also max, min, ind2sub, sub2ind.

%   Scott Lowe, July 2012

function [Y,Iout] = maxel(X,type)

if nargin<2
    type = 'subc';
end
[Y,I] = max(X(:));
if nargout==1; return; end;
switch lower(type);
    case 'ind'
        Iout = I;
    case 'subc'
        [Iout{1:ndims(X)}] = ind2sub(size(X),I);
    case 'sub'
        [Iout{1:ndims(X)}] = ind2sub(size(X),I);
        Iout = cell2mat(Iout);
    otherwise
        error('Bad input');
end

end