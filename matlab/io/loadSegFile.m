%LOADSEGFILE Given a raw data filename, load the data as a structure
%   LOADSEGFILE(FNAME) Loads the data in FNAME. FNAME should be a 
%   complete filename including directory.
%
%   Output has the following fields:
%
%     data               - Recording for all channels [nChn x nPnt double]
%
%     data_length_sec    - Approximate duration of recording in seconds
%
%     sampling_frequency - Datapoint sampling frequency
%
%     fs                 - Datapoint sampling frequency
%
%     channels           - Some kind of electrode naming scheme [1 x nChn cell]
%
%     sequence           - For consequtive interictal recordings, the ID of
%                          this data in the sequence
%
%     filepath           - Full filename with path
%
%     filename           - Filename without path (with extension)
%
%     segID              - ID number of the segment
%
%   Interval between datapoints is 1/sampling_frequency.
%   Actual recording duration is (nPnt-1)/sampling_frequency.

function Dat = loadSegFile(fname)

% Load the saved matfile
Dat = load(fname);
% Matfile contains a structure named the same as the file
f = fieldnames(Dat);
% Take the data out of the structure in the structure
Dat = Dat.(f{1});

% Add a shorthand for sampling frequency
Dat.fs = Dat.sampling_frequency;

% Note the file name
Dat.filepath = fname;
[~,fnm,fex] = fileparts(fname);
Dat.filename = [fnm fex];

% Use regex to find the fileID number within the filename
pat = '\w+_(\d+).mat'; % Regex format to identify the number on this file
% Apply regex
ret = regexp(Dat.filename, pat, 'tokens');
% Convert the string (in the cell in the cell) into numeric type
Dat.segID = str2double(ret{1}{1});

end