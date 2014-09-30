% SEEDRNG Seeds the MATLAB random number generator, and Shuffle.m
%    Seeds the random number generator with the number 7, so seeding
%    is predictable and results are deterministic.
%    Be careful when using this with parallel processing - you may
%    need to make sure each worker gets a different seed!
%    
%    Outputs the seeds used, so you can save them for posterity.

function [rgnseed shuffleseed] = seedrng()

% Seed MATLAB random number generator with 7
rgnseed = 7;
rng(rgnseed);

% Seed Shuffle with 7
shuffleseed = 7;
Shuffle(shuffleseed, 'seed');

end
