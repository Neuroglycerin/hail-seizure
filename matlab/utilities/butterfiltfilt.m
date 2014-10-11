%BUTTERFILTFILT Design and apply Butterworth filter.
%
% INPUTS
% ------
%   x     - Data to be filtered.
%           Filter operates on first non-singleton dimension.
%
%   fc    - Cutoffs frequencies (one frequency value for types 'lowpass'
%           and 'highpass' and two frequency values for types 'bandpass'
%           and 'bandstop').
%
%   fs    - Data sampling frequency.
%
%   order - Order of the Butterworth filter (default: order = 6)
%
%   type  - Filtering type option. Can be either 'lowpass', 'highpass',
%           'bandpass' or 'bandstop'.
%
%   direction - Direction to filter in
%               'forward' | 'backward' | {'both'}
%               NB: if this is 'both' the output has zero phase filtering,
%               and order must be an even number.
%
%  NB: If a bandpass is requested with filter bounds [0 F], a lowpass
%  filter is performed instead. If a bandpass is requested with filter
%  upper bound Inf or the Nyquist freq, a highpass filter is performed.
%  
%
% OUTPUTS
% -------
%   x     - Filtered data
%
%   A     - Filter coefficient matrix (denominator)
%
%   B     - Filter coefficient matrix (numerator)

function [x, A, B] = butterfiltfilt(x, fc, fs, order, type, direction)

% DEFAULT INPUTS ----------------------------------------------------------
if nargin<6 || isempty(direction)
    direction = 'both';
end
if nargin<5 || isempty(type)
    type = 'bandpass';
end
if nargin<4 || isempty(order)
    order = 6;
end
if nargin<3 || isempty(fc)
    if all(fc>=0) && all(fc<=1)
        fs = 1;
    else
        error('Sampling frequency not given');
    end
end

% INPUT HANDLING ----------------------------------------------------------
if nargin<2; error('Insufficient number of inputs'); end
if ~isnumeric(x);  error('Data must be numeric'); end;
if ~isnumeric(fc); error('Filter bounds must be numeric'); end
if ~isnumeric(fs) || ~isscalar(fs); error('Sampling frequency must be a scalar'); end
if ~isnumeric(order) || ~isscalar(order); error('Order must be a scalar'); end
if ~ischar(type); error('Type must be a string'); end
if ~ischar(direction); error('Direction must be a string'); end

% MAIN --------------------------------------------------------------------

% If bidirectional filter, halve the order as filter is done twice
if strcmp(direction,'both')
    if mod(order,2)~=0;
        error('Filter order must be even if filter is bidirectional');
    end
    order = order/2;
end
% Change type to lowercase
type = lower(type);

% Move to band notation
switch type
    % Lowpass
    case {'low','lowpass'}
        if numel(fc)~=1; error('Too many bounds given'); end;
        fc = [0 fc];
        
    % Highpass
    case {'high','highpass'}
        if numel(fc)~=1; error('Too many bounds given'); end;
        fc = [fc Inf];
        
    % Bandpass
    case {'band','bandpass'}
        % Do nothing
        
    % Bandstop
    case {'bandstop','stop'}
        type = 'stop'; % Cannonical
        
    otherwise
        error('Type %s not found',type);
        
end

% Make bounds be relative to Nyquist frequency
fc = fc*2/fs;

% Design the filter -------------------------------------------------------
if numel(fc)~=2
    error('Filter requires two cutoff frequencies');
    
elseif strcmp(type,'stop')
    % Do bandstop filter
    [B,A] = butter(order, fc, 'stop');
    % If not bandstop, must be bandpass.
    
elseif (fc(1)==0) && (isinf(fc(2)))
    % No filter required
    B = NaN;
    A = NaN;
    return;
    
elseif fc(1)==0
    % Lower bound is 0: design lowpass filter
    [B,A] = butter(order, fc(2), 'low');
    
elseif isinf(fc(2)) || fc(2)==1
    % Upper bound is Inf or Nyquist freq: design highpass filter
    [B,A] = butter(order, fc(1), 'high');
    
else
    [B,A] = butter(order, fc);
    
end
        
% Use the filter ----------------------------------------------------------
switch direction
    case 'forward'
        x = FilterM(B, A, x);
    case 'backward'
        x = FilterM(B, A, x, [], 'reverse');
    case 'both'
        if ispc
            x = filter(B, A, x);
            x(end:-1:1,:) = filter(B, A, x(end:-1:1,:));
        else
            x = FiltFiltM(B, A, x);
        end
    otherwise
        error('Unrecognised filter direction: %s',direction);
end

end