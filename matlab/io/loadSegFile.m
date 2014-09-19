% LOADSEGFILE Given a raw data filename, load the data as a structure
% Filename should include directory
% Output has the following fields:
%     data: Recording for all channels [nChn x nPnt double]
%     data_length_sec: Approximate duration of recording in seconds
%     sampling_frequency: Datapoint sampling frequency
%     channels: Some kind of electrode naming scheme [1xnChn cell]
%     sequence: For consequtive interictal recordings, the ID of this data
%               in the sequence
% Interval between datapoints is 1/sampling_frequency.
% Actual recording duration is (nPnt-1)/sampling_frequency.

function Dat = loadSegFile(fname)

Dat = load(fname);
f = fieldnames(Dat);
Dat = Dat.(f{1});

end