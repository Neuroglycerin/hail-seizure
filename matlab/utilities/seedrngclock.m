% SEEDRNG Seeds the MATLAB random number generator, and Shuffle.m
%    Seeds using the clock, so seeding is unpredictable
%    and will be different on each run.
%    Seed changes every millisecond (the accuracy with which MATLAB
%    knows the time).
%    Be careful when using this with parallel processing - make sure
%    each worker gets a different seed!
%    
%    Outputs the seeds used, so you can save them for posterity.

function [rgnseed shuffleseed] = seedrng()

% Seed MATLAB random number generator with clock time
% Clock is accurate to nearest millisecond, so only changes every ms
% Sample at this accuracy and map to [0,2^32) - the range of rng.
% Seed numbers repeat in a cycle lasting ~48 days.
rgnseed = mod(floor(now*10^8),2^32);
rng(rgnseed);

% Use MATLAB random number generator, to make a seed for Shuffle
shuffleseed = randi(2^32,[1 4])-1; % Generate a seed for Shuffle
Shuffle(shuffleseed, 'seed');  % Seed Shuffle

% Reset the MATLAB random number seed to be the one we are returning
rng(rgnseed);

end
