
% HANNING.M     - returns the N-point Hanning window in a column vector.
%
% Input		: n = signal length (or column vector?) 
% Output	: N-point Hanning window
%
% Usage		: w = hanning(n);  or 2D:  hanning(n1,n2);  
%
% Comments	: empty argument shows plot, modification of original MATLAB M-file      

% HGFei 

function w = hanning(n,n2)

if nargin == 2;  
    w1 = hanning(n);
    w2 = hanning(n2);
    w = w1(:) * w2;
    return;
end; 
    
if nargin == 0
 help hanning;
 plot(hanning(144)); 
 title(' plot(hanning(144));  '); 
 text(75,.5,'this is hanning(144)'); 
 figure(gcf); 
 return;
end;

[h,w] = size(n);
if h*w > 1;
n = max(h,w);
end; 
 
u = .5*(1 - cos(2*pi*(0:n-1)'/(n-1)));

if h*w > 1;
w = zeros(h,w);
w(:) = u; 
else
w = u(:).';    
end;  

global ROWCOLMODE 

if  isempty('ROWCOLMODE')  == 0; 
    if  strcmp(ROWCOLMODE,'cols') == 1; 
        w =  w.';
    end;
end;