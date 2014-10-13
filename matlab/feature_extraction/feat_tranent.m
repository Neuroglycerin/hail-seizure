%function feat_tranent()

addpath(genpathexcept('matlab',{'TRENTOOL2'},{'OpenTSTOOL','TRENTOOL3','fieldtrip-20140929'}))


[fnames, mydir, segIDs] = subjtyp2dirs('Dog_1', 'inter');
Dat = loadSegFile(fullfile(mydir,fnames{1}));


ft_defaults

[nChn, nPnt] = size(Dat.data);

% Convert to FieldTrip format
% Initialise
data = [ ];
data.trial = {Dat.data};
% Add time series into .time
data.time    = {(0:nPnt-1)/Dat.fs};
% Add channel labels into .label
data.label   = {};
data.label   = cell(1,nChn);
for iChn=1:nChn
    data.label{1,iChn} = num2str(iChn);
end
% Add sampling frequency into .fsample
data.fsample = Dat.fs;


% TEprepare
cfg = [ ];

cfg.trialselect = 'no';

cfg.channel = data.label; % do all channel pairs
cfg.Path2TSTOOL = fullfile(getRepoDir(), 'matlab', 'dependencies', 'OpenTSTOOL');

% use the whole trial for analysis
cfg.toi = [min(data.time{1}), max(data.time{1})]; % time of interest

% scanning of interaction delays u
cfg.predicttime_u = 50;   % minimum u to be scanned

% optimizing embedding
cfg.optimizemethod = 'ragwitz'; % criterion used
cfg.ragdim = 2:9; % criterion dimension
cfg.ragtaurange = [0.2 0.4]; % range for tau
cfg.ragtausteps = 5; % steps for ragwitz tau steps
cfg.repPred = 100; % size ( data.trial {1 ,1} ,2) *(3/4) ;
cfg.flagNei = 'Mass';
cfg.sizeNei = 4;

% estimator
cfg.TEcalctype = 'VW_ds'; % use the new TE estimator (Wibral , 2013)

% ACT estimation and constraints on allowed ACT (auto corelation time)
cfg.actthrvalue = 100000000; % threshold for ACT
cfg.maxlag      = 1000;
cfg.minnrtrials = 1; % minimum acceptable number of trials

Prepared_Data = TEprepare(cfg,data)


%% EXAMPLE from presentation
% cfg.Path2TSTOOL = '~/user01_vnc28/toolboxes/TRENTOOL2';
% cfg.toi = [0.001 3];
% cfg.channel = data.label;
% cfg.predicttime_u = 46;
% cfg.actthrvalue = 50;
% cfg.minnrtrials = 12;
% cfg.optimizemethod = 'ragwitz';
% cfg.ragdim = 2:8;
% cfg.ragtaurange = [0.5 1];
% cfg.ragtausteps = 15;
% cfg.repPred = 1000;
% cfg.flagNei = 'Mass';
% cfg.sizeNei = 4;

% %save('~/user01_vnc28/data/data_save/Prepared_Data.mat','Prepared_Data')
% % TEsurrogatestats
% cfg = [ ];
% cfg.optdimusage = 'indivdim';
% cfg.tail = 1;
% cfg.numpermutation = 50000;
% cfg.shifttesttype = 'TEshift>TE';
% cfg.surrogatetype = 'trialshuffling';
% cfg.fileidout = '~/user01_vnc28/data/data_save/TEsur_output';

% TEsurrogatestats(cfg,Prepared_Data)


%% EXAMPLE from webpage
% filedtripdefs;
% 
% % load the data
% load(datafilename)); % say this contains the variable "data"
% 
% %%% TE prepare settings %%%
% cfg=[];
% cfg.optimizemethod='cao'; % select the CAO method to estimate embedding dimensions
% cfg.toi = [-0.85 0.05]; % the time interval of interest in secs 
% cfg.predicttime_u = 32; % the prediction time im ms (e.g. the axonal delay; always choose bigger values when in doubt - also see the Vicente, 2010 paper)
% cfg.channel = data.label; % a cell array with the names of the channles you want to analyze. here all cahnnels in the dataset are analyzed
% % Alternative: use cfg.sgncmb to just analyze specific signal combinations e.g. from a channel named channel1 to a  channel named channel101, 
% % and from a channel named channelA to a  channel named channelB :
% % cfg.sgncmb={'channel1', 'channel101'; 
% %                       'channelA', 'channelB'} 
% cfg.caodim = 1:8; % check dimensions between 1 and 8;
% cfg.caokth_neighbors = 3;
% cfg.actthrvalue=120; % Do not accpet trials that have a longer autocorrelation decay time (ACT) than this (helps to avoid 'out of data' errors)
% cfg.minnrtrials=10; % abort calculations if we do not have a least this number of trials with acceptable ACT
% cfg.Path2TSTOOL = '/data/common/OpenTSTOOL_v1-2'; % the path where you unpacked the TStool toolbox
% cfg.feedback = 'no'; % ...
% data=TEprepare(cfg,data); % the actual command that prepares the data for subsequent TE analysis
% 
% %%% stats section %%%
% cfg = [];
% cfg.surrogatetype = 'trialshuffling'; % the type of surrogates that are created for significance testing
% cfg.shifttesttype='TEshift>TE'; % the type of shift test that is used (Vicente, J Comp Neursci, 2010 and Wibral, Prog Biophys Mol Biol, 2010)
% cfg.fileidout = strcat('My_prefix_'); % a prefix for the results filename
% TEsurrogatestats(cfg,data); % the actual command that computes transfer entropy for the original data, the surrogate data, and the shifttest data and that performs the statistical test.  


%% EXAMPLE from doc

% % % define cfg for TEprepare.m
% 
% cfgTEP = [];
% 
% % path to OpenTSTOOL
% cfgTEP.Path2TSTOOL = '../ OpenTSTOOL ';
% 
% % data
% cfgTEP.toi = [min(data.time{1 ,1}), max(data.time{1 ,1})]; % time of interest
% cfgTEP.sgncmb = {'A1' 'A2'}; % channels to be analyzed
% 
% % scanning of interaction delays u
% cfgTEP.predicttimemin_u = 40;   % minimum u to be scanned
% cfgTEP.predicttimemax_u = 50;   % maximum u to be scanned
% cfgTEP.predicttimestepsize = 1; % time steps between u's to be scanned
% 
% % estimator
% cfgTEP.TEcalctype = 'VW_ds'; % use the new TE estimator ( Wibral , 2013)
% 
% % ACT estimation and constraints on allowed ACT ( auto corelati on time )
% cfgTEP.actthrvalue = 100; % threshold for ACT
% cfgTEP.maxlag      = 1000;
% cfgTEP.minnrtrials = 15; % minimum acceptable number of trials
% 
% % optimizing embedding
% cfgTEP.optimizemethod = 'ragwitz'; % criterion used
% cfgTEP.ragdim = 2:9; % criterion dimension
% cfgTEP.ragtaurange = [0.2 0.4]; % range for tau
% cfgTEP.ragtausteps = 5; % steps for ragwitz tau steps
% cfgTEP.repPred = 100; % size ( data.trial {1 ,1} ,2) *(3/4) ;
% 
% % kernel - based TE estimation
% cfgTEP.flagNei = 'Mass' ; % neigbour analyse type


%end