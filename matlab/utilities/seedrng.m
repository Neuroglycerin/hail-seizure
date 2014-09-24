% SEEDRNG Seeds the MATLAB random number generator, and Shuffle.m
%    Seeds using the clock, so seeding is unpredictable
%    and will be different on each run.
%    Be careful when using this with parallel processing - make sure
%    each worker gets a different seed!
%    Outputs the seeds used, incase you need to save them for posterity.

function [rgnseed shuffleseed] = seedrng()

% Seed MATLAB random number generator with clock time
rgnseed = sum(100*clock);
rng(rgnseed);

% Use MATLAB random number generator, to make a seed for Shuffle
shuffleseed = randi(2^32,[1 4])-1; % Generate a seed for Shuffle
Shuffle(shuffleseed, 'seed');  % Seed Shuffle

end
