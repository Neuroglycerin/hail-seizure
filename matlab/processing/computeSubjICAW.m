% Computes and saves the ICA weights for a subject
% Uses all datapoints in all datafiles for the subject
% FastICA functions were taken and copied here as nested functions with
% shared variables to reduce memory consumption.
function W = computeSubjICAW(subj, dbgmde, do_icafiles)

% =========================================================================
% Initialise random number generator
rng(7);

% =========================================================================
% Default inputs ----------------------------------------------------------
if nargin<3;
    do_icafiles = false; % Whether to prevent overwritting files
end
if nargin<2;
    dbgmde   = true; % Whether to write progress to screen
end
pltmde = 'off'; % Whether to plot ICA progress

% Parameters --------------------------------------------------------------
ictypall = {'inter','pre','test'}; % Which ictypes exist
ictypuse = {'inter','pre','test'}; % Which ictypes to use for the ICA model

% Input Handling ----------------------------------------------------------
% Check subject is one for which there is data
if ~ismember(subj,subjnames())
    error('Bad subject name: %s',subj);
end

% Main --------------------------------------------------------------------

if dbgmde; fprintf('Performing ICA processing for subject %s\n',subj); end

% Find ICA model ----------------------------------------------------------
% Get a list of all the filenames in the ictypes we will use --------------
fnamelist = {};
for iTyp=1:length(ictypuse)
    icfnamelist = dircat(subj, ictypuse{iTyp});
    fnamelist = [fnamelist(:); icfnamelist(:)];
end

% Load all the data and put it into an enormous matrix --------------------
nFle = length(fnamelist);
fleSrtIdx = nan(1,nFle+1);
fleSrtIdx(1) = 1;
if dbgmde; fprintf('Loading %d raw data files\n',nFle); end
for iFle=1:nFle
    if dbgmde && (mod(iFle,50)==0 || iFle==1 || iFle==2);
        fprintf('Loading file %3d/%3d\n',iFle,nFle);
    end
    % Load the saved matfile
    Dat = load(fnamelist{iFle});
    % Matfile contains a structure named the same as the file
    f = fieldnames(Dat);
    % Check how many datapoints there are
    datlen = size(Dat.(f{1}).data,2);
    % Initialise the holding matrix
    if iFle==1
        if dbgmde; fprintf('Making RAM available for large dataset\n'); end
        mixedsig = nan(size(Dat.(f{1}).data,1), datlen*nFle);
    end
    % Take the data out of the structure in the structure
    mixedsig(:,fleSrtIdx(iFle)+(0:datlen-1)) = Dat.(f{1}).data;
    % Note where the next file should start its entry
    fleSrtIdx(iFle+1) = fleSrtIdx(iFle) + datlen;
end

% Cut off any excess length
if size(mixedsig,2)~=fleSrtIdx(end)-1
    if dbgmde; fprintf('Cropping dataset appropriately\n'); end
    % mixedsig = mixedsig(:,1:fleSrtIdx(end)-1);
    mixedsig(:,fleSrtIdx(end):end) = [];
end

% =========================================================================
% Perform ICA on the enormous dataset -------------------------------------
if dbgmde; fprintf('Running FastICA\n'); end
[~, A, W] = fastica('displayMode', pltmde);
clear mixedsig;
% =========================================================================

% Write the ICA matrix to file --------------------------------------------
[Wfname,Wfnamefull_log] = getWfname(subj);
if dbgmde; fprintf('Writing weight matrix to file\n  %s\n',Wfname); end
if ~exist(fileparts(Wfname),'dir'); mkdir(fileparts(Wfname)); end;
save(Wfname,'W','A'); % Overwrite the copy to be used in transformations
if ~exist(fileparts(Wfnamefull_log),'dir'); mkdir(fileparts(Wfnamefull_log)); end;
save(Wfnamefull_log,'W','A'); % Save a dated copy for posterity

% Quit if we are not writing any files
if ~do_icafiles; return; end;

% Process each file -------------------------------------------------------
% Get a list of all the filenames in all the ictypes ----------------------
fnamelist = {};
for iTyp=1:length(ictypall)
    icfnamelist = dircat(subj, ictypall{iTyp});
    fnamelist = [fnamelist(:); icfnamelist(:)];
end

% For each datafile, load it up, do the separating matrix transformation,
% and save as a different file
nFle = length(fnamelist);
for iFle=1:nFle
    if dbgmde && mod(iFle,100)==0; fprintf('Processing file %3d/%3d\n',iFle,nFle); end
    % Load the saved matfile
    Dat = load(fnamelist{iFle});
    % Matfile contains a structure named the same as the file
    f = fieldnames(Dat);
    % Transform the data in the structure in the structure
    Dat.(f{1}).data = W * Dat.(f{1}).data;
    % Get new file name
    icafname = raw2icafname(fnamelist{iFle});
    % Make directory if necessary
    if ~exist(fileparts(icafname),'dir'); mkdir(fileparts(icafname)); end;
    % Save ICA
    save(icafname,'-v7.3','-struct','Dat');
end

if dbgmde; fprintf('Finished ICA processing for subject %s\n',subj); end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN rip from FastICA
% Taken from FastICA_25 toolbox, moved here to prevent variable duplication
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [Out1, Out2, Out3] = fastica(varargin)
        %FASTICA - Fast Independent Component Analysis
        %
        % FastICA for Matlab 7.x and 6.x
        % Version 2.5, October 19 2005
        % Copyright (c) Hugo G�vert, Jarmo Hurri, Jaakko S�rel�, and Aapo Hyv�rinen.
        
        % @(#)$Id: fastica.m,v 1.14 2005/10/19 13:05:34 jarmo Exp $
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check some basic requirements of the data
        if nargin == 0,
            error ('You must supply the mixed data as input argument.');
        end
        
        if length (size (mixedsig)) > 2,
            error ('Input data can not have more than two dimensions.');
        end
        
        if any (any (isnan (mixedsig))),
            error ('Input data contains NaN''s.');
        end
        
        if ~isa (mixedsig, 'double')
            fprintf ('Warning: converting input data into regular (double) precision.\n');
            mixedsig = double (mixedsig);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [Dim, NumOfSampl] = size(mixedsig);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Default values for optional parameters
        
        % All
        verbose           = 'on';
        
        % Default values for 'pcamat' parameters
        firstEig          = 1;
        lastEig           = Dim;
        interactivePCA    = 'off';
        
        % Default values for 'fpica' parameters
        approach          = 'defl';
        numOfIC           = Dim;
        g                 = 'pow3';
        finetune          = 'off';
        a1                = 1;
        a2                = 1;
        myy               = 1;
        stabilization     = 'off';
        epsilon           = 0.0001;
        maxNumIterations  = 1000;
        maxFinetune       = 5;
        initState         = 'rand';
        guess             = 0;
        sampleSize        = 1;
        displayMode       = 'off';
        displayInterval   = 1;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Parameters for fastICA - i.e. this file
        
        b_verbose = 1;
        jumpPCA = 0;
        jumpWhitening = 0;
        only = 3;
        userNumOfIC = 0;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Read the optional parameters
        
        if (rem(length(varargin),2)==1)
            error('Optional parameters should always go by pairs');
        else
            for i=1:2:(length(varargin)-1)
                if ~ischar (varargin{i}),
                    error (['Unknown type of optional parameter name (parameter' ...
                        ' names must be strings).']);
                end
                % change the value of parameter
                switch lower (varargin{i})
                    case 'stabilization'
                        stabilization = lower (varargin{i+1});
                    case 'maxfinetune'
                        maxFinetune = varargin{i+1};
                    case 'samplesize'
                        sampleSize = varargin{i+1};
                    case 'verbose'
                        verbose = lower (varargin{i+1});
                        % silence this program also
                        if strcmp (verbose, 'off'), b_verbose = 0; end
                    case 'firsteig'
                        firstEig = varargin{i+1};
                    case 'lasteig'
                        lastEig = varargin{i+1};
                    case 'interactivepca'
                        interactivePCA = lower (varargin{i+1});
                    case 'approach'
                        approach = lower (varargin{i+1});
                    case 'numofic'
                        numOfIC = varargin{i+1};
                        % User has supplied new value for numOfIC.
                        % We'll use this information later on...
                        userNumOfIC = 1;
                    case 'g'
                        g = lower (varargin{i+1});
                    case 'finetune'
                        finetune = lower (varargin{i+1});
                    case 'a1'
                        a1 = varargin{i+1};
                    case 'a2'
                        a2 = varargin{i+1};
                    case {'mu', 'myy'}
                        myy = varargin{i+1};
                    case 'epsilon'
                        epsilon = varargin{i+1};
                    case 'maxnumiterations'
                        maxNumIterations = varargin{i+1};
                    case 'initguess'
                        % no use setting 'guess' if the 'initState' is not set
                        initState = 'guess';
                        guess = varargin{i+1};
                    case 'displaymode'
                        displayMode = lower (varargin{i+1});
                    case 'displayinterval'
                        displayInterval = varargin{i+1};
                    case 'pcae'
                        % calculate if there are enought parameters to skip PCA
                        jumpPCA = jumpPCA + 1;
                        E = varargin{i+1};
                    case 'pcad'
                        % calculate if there are enought parameters to skip PCA
                        jumpPCA = jumpPCA + 1;
                        D = varargin{i+1};
                    case 'whitesig'
                        % calculate if there are enought parameters to skip PCA and whitening
                        jumpWhitening = jumpWhitening + 1;
                        whitesig = varargin{i+1};
                    case 'whitemat'
                        % calculate if there are enought parameters to skip PCA and whitening
                        jumpWhitening = jumpWhitening + 1;
                        whiteningMatrix = varargin{i+1};
                    case 'dewhitemat'
                        % calculate if there are enought parameters to skip PCA and whitening
                        jumpWhitening = jumpWhitening + 1;
                        dewhiteningMatrix = varargin{i+1};
                    case 'only'
                        % if the user only wants to calculate PCA or...
                        switch lower (varargin{i+1})
                            case 'pca'
                                only = 1;
                            case 'white'
                                only = 2;
                            case 'all'
                                only = 3;
                        end
                        
                    otherwise
                        % Hmmm, something wrong with the parameter string
                        error(['Unrecognized parameter: ''' varargin{i} '''']);
                end;
            end;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % print information about data
        if b_verbose
            fprintf('Number of signals: %d\n', Dim);
            fprintf('Number of samples: %d\n', NumOfSampl);
        end
        
        % Check if the data has been entered the wrong way,
        % but warn only... it may be on purpose
        
        if Dim > NumOfSampl
            if b_verbose
                fprintf('Warning: ');
                fprintf('The signal matrix may be oriented in the wrong way.\n');
                fprintf('In that case transpose the matrix.\n\n');
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Remove the mean
        
        if b_verbose; disp('Removing mean...'); end;
        % Remove mean (remmean.m)
        % [mixedsig, mixedmean] = remmean(mixedsig);
        % mixedmean = mean(mixedsig,2);
        mixedmean = sum(mixedsig,2)/size(mixedsig,2);
        % mixedsig = mixedsig - mixedmean * ones(1, size(mixedsig,2));
        % mixedsig = bsxfun(@minus, mixedsig, mixedmean); % More memory, but a bit faster
        for iPnt=1:size(mixedsig,2) % Less memory, but a bit slow
            mixedsig(:,iPnt) = mixedsig(:,iPnt) - mixedmean;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate PCA
        [E, D]=pcamat(NaN, firstEig, lastEig, interactivePCA, verbose);
        
        % Calculate the whitening
        % From whitenv.m
        % ========================================================
        % In some cases, rounding errors in Matlab cause negative
        % eigenvalues (elements in the diagonal of D). Since it
        % is difficult to know when this happens, it is difficult
        % to correct it automatically. Therefore an error is
        % signalled and the correction is left to the user.
        if any (diag (D) < 0),
            error (sprintf (['[ %d ] negative eigenvalues computed from the' ...
                ' covariance matrix.\nThese are due to rounding' ...
                ' errors in Matlab (the correct eigenvalues are\n' ...
                'probably very small).\nTo correct the situation,' ...
                ' please reduce the number of dimensions in the' ...
                ' data\nby using the ''lastEig'' argument in' ...
                ' function FASTICA, or ''Reduce dim.'' button\nin' ...
                ' the graphical user interface.'], ...
                sum (diag (D) < 0)));
        end
        
        % ========================================================
        % Calculate the whitening and dewhitening matrices (these handle
        % dimensionality simultaneously).
        whiteningMatrix = inv (sqrt (D)) * E';
        dewhiteningMatrix = E * sqrt (D);
        
        % Project to the eigenvectors of the covariance matrix.
        % Whiten the samples and reduce dimension simultaneously.
        if b_verbose, fprintf ('Whitening...\n'); end
        mixedsig =  whiteningMatrix * mixedsig; % More memory, but much fast
        %for iPnt=1:size(mixedsig,2) % Less memory, but much slower
        %    mixedsig(:,iPnt) =  whiteningMatrix * mixedsig(:,iPnt);
        %end
        
        % ========================================================
        % Just some security...
        if ~isreal(mixedsig)
            error ('Whitened vectors have imaginary values.');
        end
        
        % Print some information to user
        %         if b_verbose
        %             fprintf ('Check: covariance differs from identity by [ %g ].\n', ...
        %                 max (max (abs (cov (mixedsig', 1) - eye (size (mixedsig, 1))))));
        %         end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculating the ICA
        
        % Check some parameters
        % The dimension of the data may have been reduced during PCA calculations.
        % The original dimension is calculated from the data by default, and the
        % number of IC is by default set to equal that dimension.
        
        Dim = size(mixedsig, 1);
        
        % The number of IC's must be less or equal to the dimension of data
        if numOfIC > Dim
            numOfIC = Dim;
            % Show warning only if verbose = 'on' and user supplied a value for 'numOfIC'
            if (b_verbose & userNumOfIC)
                fprintf('Warning: estimating only %d independent components\n', numOfIC);
                fprintf('(Can''t estimate more independent components than dimension of data)\n');
            end
        end
        
        % Calculate the ICA with fixed point algorithm.
        [A, W] = fpica (NaN,  whiteningMatrix, dewhiteningMatrix, approach, ...
            numOfIC, g, finetune, a1, a2, myy, stabilization, epsilon, ...
            maxNumIterations, maxFinetune, initState, guess, sampleSize, ...
            displayMode, displayInterval, verbose);
        
        % Check for valid return
        if ~isempty(W)
            % Add the mean back in.
            %if b_verbose
            %    fprintf('Adding the mean back to the data.\n');
            %end
            %icasig = W * mixedsig + (W * mixedmean) * ones(1, NumOfSampl);
            icasig = [];
            %icasig = W * mixedsig;
            if b_verbose & ...
                    (max(abs(W * mixedmean)) > 1e-9) & ...
                    (strcmp(displayMode,'signals') | strcmp(displayMode,'on'))
                fprintf('Note that the plots don''t have the mean added.\n');
            end
        else
            icasig = [];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The output depends on the number of output parameters
        % and the 'only' parameter.
        
        if only == 1    % only PCA
            Out1 = E;
            Out2 = D;
        elseif only == 2  % only PCA & whitening
            if nargout == 2
                Out1 = whiteningMatrix;
                Out2 = dewhiteningMatrix;
            else
                Out1 = mixedsig;
                Out2 = whiteningMatrix;
                Out3 = dewhiteningMatrix;
            end
        else      % ICA
            if nargout == 2
                Out1 = A;
                Out2 = W;
            else
                Out1 = icasig;
                Out2 = A;
                Out3 = W;
            end
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [E, D] = pcamat(ignore, firstEig, lastEig, s_interactive, ...
                s_verbose)
            %PCAMAT - Calculates the pca for data
            
            
            % @(#)$Id: pcamat.m,v 1.5 2003/12/15 18:24:32 jarmo Exp $
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Default values:
            if nargin < 5, s_verbose = 'on'; end
            if nargin < 4, s_interactive = 'off'; end
            if nargin < 3, lastEig = size(mixedsig, 1); end
            if nargin < 2, firstEig = 1; end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Check the optional parameters;
            switch lower(s_verbose)
                case 'on'
                    b_verbose = 1;
                case 'off'
                    b_verbose = 0;
                otherwise
                    error(sprintf('Illegal value [ %s ] for parameter: ''verbose''\n', s_verbose));
            end
            
            switch lower(s_interactive)
                case 'on'
                    b_interactive = 1;
                case 'off'
                    b_interactive = 0;
                case 'gui'
                    b_interactive = 2;
                otherwise
                    error(sprintf('Illegal value [ %s ] for parameter: ''interactive''\n', ...
                        s_interactive));
            end
            
            oldDimension = size (mixedsig, 1);
            if ~(b_interactive)
                if lastEig < 1 | lastEig > oldDimension
                    error(sprintf('Illegal value [ %d ] for parameter: ''lastEig''\n', lastEig));
                end
                if firstEig < 1 | firstEig > lastEig
                    error(sprintf('Illegal value [ %d ] for parameter: ''firstEig''\n', firstEig));
                end
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Calculate PCA
            
            % Calculate the covariance matrix.
            if b_verbose, fprintf ('Calculating covariance...\n'); end
            % covarianceMatrix = cov(mixedsig', 1);
            % (Already removed the mean, so no need to repeat this step)
            %covarianceMatrix = (mixedsig * mixedsig') / size(mixedsig,2);
            [nX,nY] = size(mixedsig);
            covarianceMatrix = nan(nX,nX);
            for x2=1:nX
                if b_verbose, fprintf('... %d/%d\n',x2,nX); end
                covarianceMatrix(:,x2) = (mixedsig * mixedsig(x2,:)') / nY;
            end
            %covarianceMatrix = zeros(nX,nX);
            %for y=1:nY
            %    if b_verbose, fprintf('... %d/%d\n',y,nY); end
            %    covarianceMatrix = covarianceMatrix + (mixedsig(:,y) * mixedsig(:,y)');
            %end
            %covarianceMatrix = covarianceMatrix / nY;
            
            % Calculate the eigenvalues and eigenvectors of covariance
            % matrix.
            if b_verbose, fprintf ('Calculating eigenvalues of covariance matrix...\n'); end
            [E, D] = eig (covarianceMatrix);
            
            % The rank is determined from the eigenvalues - and not directly by
            % using the function rank - because function rank uses svd, which
            % in some cases gives a higher dimensionality than what can be used
            % with eig later on (eig then gives negative eigenvalues).
            rankTolerance = 1e-7;
            maxLastEig = sum (diag (D) > rankTolerance);
            if maxLastEig == 0,
                fprintf (['Eigenvalues of the covariance matrix are' ...
                    ' all smaller than tolerance [ %g ].\n' ...
                    'Please make sure that your data matrix contains' ...
                    ' nonzero values.\nIf the values are very small,' ...
                    ' try rescaling the data matrix.\n'], rankTolerance);
                error ('Unable to continue, aborting.');
            end
            
            % Sort the eigenvalues - decending.
            eigenvalues = flipud(sort(diag(D)));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % See if the user has reduced the dimension enought
            
            if lastEig > maxLastEig
                lastEig = maxLastEig;
                if b_verbose
                    fprintf('Dimension reduced to %d due to the singularity of covariance matrix\n',...
                        lastEig-firstEig+1);
                end
            else
                % Reduce the dimensionality of the problem.
                if b_verbose
                    if oldDimension == (lastEig - firstEig + 1)
                        fprintf ('Dimension not reduced.\n');
                    else
                        fprintf ('Reducing dimension...\n');
                    end
                end
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop the smaller eigenvalues
            if lastEig < oldDimension
                lowerLimitValue = (eigenvalues(lastEig) + eigenvalues(lastEig + 1)) / 2;
            else
                lowerLimitValue = eigenvalues(oldDimension) - 1;
            end
            
            lowerColumns = diag(D) > lowerLimitValue;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop the larger eigenvalues
            if firstEig > 1
                higherLimitValue = (eigenvalues(firstEig - 1) + eigenvalues(firstEig)) / 2;
            else
                higherLimitValue = eigenvalues(1) + 1;
            end
            higherColumns = diag(D) < higherLimitValue;
            
            % Combine the results from above
            selectedColumns = lowerColumns & higherColumns;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % print some info for the user
            if b_verbose
                fprintf ('Selected [ %d ] dimensions.\n', sum (selectedColumns));
            end
            if sum (selectedColumns) ~= (lastEig - firstEig + 1),
                error ('Selected a wrong number of dimensions.');
            end
            
            if b_verbose
                fprintf ('Smallest remaining (non-zero) eigenvalue [ %g ]\n', eigenvalues(lastEig));
                fprintf ('Largest remaining (non-zero) eigenvalue [ %g ]\n', eigenvalues(firstEig));
                fprintf ('Sum of removed eigenvalues [ %g ]\n', sum(diag(D) .* ...
                    (~selectedColumns)));
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Select the colums which correspond to the desired range
            % of eigenvalues.
            E = selcol(E, selectedColumns);
            D = selcol(selcol(D, selectedColumns)', selectedColumns);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Some more information
            if b_verbose
                sumAll=sum(eigenvalues);
                sumUsed=sum(diag(D));
                retained = (sumUsed / sumAll) * 100;
                fprintf('[ %g ] %% of (non-zero) eigenvalues retained.\n', retained);
            end
            
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function newMatrix = selcol(oldMatrix, maskVector)
        
        % newMatrix = selcol(oldMatrix, maskVector);
        %
        % Selects the columns of the matrix that marked by one in the given vector.
        % The maskVector is a column vector.
        
        % 15.3.1998
        
        if size(maskVector, 1) ~= size(oldMatrix, 2),
            error ('The mask vector and matrix are of uncompatible size.');
        end
        
        numTaken = 0;
        
        for ii = 1 : size (maskVector, 1),
            if maskVector(ii, 1) == 1,
                takingMask(1, numTaken + 1) = ii;
                numTaken = numTaken + 1;
            end
        end
        
        newMatrix = oldMatrix(:, takingMask);
        
    end



    function [A, W] = fpica(ignore, whiteningMatrix, dewhiteningMatrix, approach, ...
            numOfIC, g, finetune, a1, a2, myy, stabilization, ...
            epsilon, maxNumIterations, maxFinetune, initState, ...
            guess, sampleSize, displayMode, displayInterval, ...
            s_verbose)
        %FPICA - Fixed point ICA. Main algorithm of FASTICA.
        
        % @(#)$Id: fpica.m,v 1.7 2005/06/16 12:52:55 jarmo Exp $
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Global variable for stopping the ICA calculations from the GUI
        global g_FastICA_interrupt;
        if isempty(g_FastICA_interrupt)
            clear global g_FastICA_interrupt;
            interruptible = 0;
        else
            interruptible = 1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Default values
        
        if nargin < 3, error('Not enough arguments!'); end
        [vectorSize, numSamples] = size(mixedsig);
        if nargin < 20, s_verbose = 'on'; end
        if nargin < 19, displayInterval = 1; end
        if nargin < 18, displayMode = 'on'; end
        if nargin < 17, sampleSize = 1; end
        if nargin < 16, guess = 1; end
        if nargin < 15, initState = 'rand'; end
        if nargin < 14, maxFinetune = 100; end
        if nargin < 13, maxNumIterations = 1000; end
        if nargin < 12, epsilon = 0.0001; end
        if nargin < 11, stabilization = 'on'; end
        if nargin < 10, myy = 1; end
        if nargin < 9, a2 = 1; end
        if nargin < 8, a1 = 1; end
        if nargin < 7, finetune = 'off'; end
        if nargin < 6, g = 'pow3'; end
        if nargin < 5, numOfIC = vectorSize; end     % vectorSize = Dim
        if nargin < 4, approach = 'defl'; end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the data
        
        if ~isreal(mixedsig)
            error('Input has an imaginary part.');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the value for verbose
        
        switch lower(s_verbose)
            case 'on'
                b_verbose = 1;
            case 'off'
                b_verbose = 0;
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''verbose''\n', s_verbose));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the value for approach
        
        switch lower(approach)
            case 'symm'
                approachMode = 1;
            case 'defl'
                approachMode = 2;
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''approach''\n', approach));
        end
        if b_verbose, fprintf('Used approach [ %s ].\n', approach); end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the value for numOfIC
        
        if vectorSize < numOfIC
            error('Must have numOfIC <= Dimension!');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the sampleSize
        if sampleSize > 1
            sampleSize = 1;
            if b_verbose
                fprintf('Warning: Setting ''sampleSize'' to 1.\n');
            end
        elseif sampleSize < 1
            if (sampleSize * numSamples) < 1000
                sampleSize = min(1000/numSamples, 1);
                if b_verbose
                    fprintf('Warning: Setting ''sampleSize'' to %0.3f (%d samples).\n', ...
                        sampleSize, floor(sampleSize * numSamples));
                end
            end
        end
        if b_verbose
            if  b_verbose & (sampleSize < 1)
                fprintf('Using about %0.0f%% of the samples in random order in every step.\n',sampleSize*100);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the value for nonlinearity.
        
        switch lower(g)
            case 'pow3'
                gOrig = 10;
            case 'tanh'
                gOrig = 20;
            case {'gaus', 'gauss'}
                gOrig = 30;
            case 'skew'
                gOrig = 40;
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''g''\n', g));
        end
        if sampleSize ~= 1
            gOrig = gOrig + 2;
        end
        if myy ~= 1
            gOrig = gOrig + 1;
        end
        
        if b_verbose,
            fprintf('Used nonlinearity [ %s ].\n', g);
        end
        
        finetuningEnabled = 1;
        switch lower(finetune)
            case 'pow3'
                gFine = 10 + 1;
            case 'tanh'
                gFine = 20 + 1;
            case {'gaus', 'gauss'}
                gFine = 30 + 1;
            case 'skew'
                gFine = 40 + 1;
            case 'off'
                if myy ~= 1
                    gFine = gOrig;
                else
                    gFine = gOrig + 1;
                end
                finetuningEnabled = 0;
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''finetune''\n', ...
                    finetune));
        end
        
        if b_verbose & finetuningEnabled
            fprintf('Finetuning enabled (nonlinearity: [ %s ]).\n', finetune);
        end
        
        switch lower(stabilization)
            case 'on'
                stabilizationEnabled = 1;
            case 'off'
                if myy ~= 1
                    stabilizationEnabled = 1;
                else
                    stabilizationEnabled = 0;
                end
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''stabilization''\n', ...
                    stabilization));
        end
        
        if b_verbose & stabilizationEnabled
            fprintf('Using stabilized algorithm.\n');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Some other parameters
        myyOrig = myy;
        % When we start fine-tuning we'll set myy = myyK * myy
        myyK = 0.01;
        % How many times do we try for convergence until we give up.
        failureLimit = 5;
        
        
        usedNlinearity = gOrig;
        stroke = 0;
        notFine = 1;
        long = 0;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the value for initial state.
        
        switch lower(initState)
            case 'rand'
                initialStateMode = 0;
            case 'guess'
                if size(guess,1) ~= size(whiteningMatrix,2)
                    initialStateMode = 0;
                    if b_verbose
                        fprintf('Warning: size of initial guess is incorrect. Using random initial guess.\n');
                    end
                else
                    initialStateMode = 1;
                    if size(guess,2) < numOfIC
                        if b_verbose
                            fprintf('Warning: initial guess only for first %d components. Using random initial guess for others.\n', size(guess,2));
                        end
                        guess(:, size(guess, 2) + 1:numOfIC) = ...
                            rand(vectorSize,numOfIC-size(guess,2))-.5;
                    elseif size(guess,2)>numOfIC
                        guess=guess(:,1:numOfIC);
                        fprintf('Warning: Initial guess too large. The excess column are dropped.\n');
                    end
                    if b_verbose, fprintf('Using initial guess.\n'); end
                end
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''initState''\n', initState));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking the value for display mode.
        
        switch lower(displayMode)
            case {'off', 'none'}
                usedDisplay = 0;
            case {'on', 'signals'}
                usedDisplay = 1;
                if (b_verbose & (numSamples > 10000))
                    fprintf('Warning: Data vectors are very long. Plotting may take long time.\n');
                end
                if (b_verbose & (numOfIC > 25))
                    fprintf('Warning: There are too many signals to plot. Plot may not look good.\n');
                end
            case 'basis'
                usedDisplay = 2;
                if (b_verbose & (numOfIC > 25))
                    fprintf('Warning: There are too many signals to plot. Plot may not look good.\n');
                end
            case 'filters'
                usedDisplay = 3;
                if (b_verbose & (vectorSize > 25))
                    fprintf('Warning: There are too many signals to plot. Plot may not look good.\n');
                end
            otherwise
                error(sprintf('Illegal value [ %s ] for parameter: ''displayMode''\n', displayMode));
        end
        
        % The displayInterval can't be less than 1...
        if displayInterval < 1
            displayInterval = 1;
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if b_verbose, fprintf('Starting ICA calculation...\n'); end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SYMMETRIC APPROACH
        if approachMode == 1,
            
            % set some parameters more...
            usedNlinearity = gOrig;
            stroke = 0;
            notFine = 1;
            long = 0;
            
            A = zeros(vectorSize, numOfIC);  % Dewhitened basis vectors.
            if initialStateMode == 0
                % Take random orthonormal initial vectors.
                B = orth (randn (vectorSize, numOfIC));
            elseif initialStateMode == 1
                % Use the given initial vector as the initial state
                B = whiteningMatrix * guess;
            end
            
            BOld = zeros(size(B));
            BOld2 = zeros(size(B));
            
            % This is the actual fixed-point iteration loop.
            for round = 1:maxNumIterations + 1,
                if round == maxNumIterations + 1,
                    fprintf('No convergence after %d steps\n', maxNumIterations);
                    fprintf('Note that the plots are probably wrong.\n');
                    if ~isempty(B)
                        % Symmetric orthogonalization.
                        B = B * real(inv(B' * B)^(1/2));
                        
                        W = B' * whiteningMatrix;
                        A = dewhiteningMatrix * B;
                    else
                        W = [];
                        A = [];
                    end
                    return;
                end
                
                if (interruptible & g_FastICA_interrupt)
                    if b_verbose
                        fprintf('\n\nCalculation interrupted by the user\n');
                    end
                    if ~isempty(B)
                        W = B' * whiteningMatrix;
                        A = dewhiteningMatrix * B;
                    else
                        W = [];
                        A = [];
                    end
                    return;
                end
                
                
                % Symmetric orthogonalization.
                B = B * real(inv(B' * B)^(1/2));
                
                % Test for termination condition. Note that we consider opposite
                % directions here as well.
                minAbsCos = min(abs(diag(B' * BOld)));
                minAbsCos2 = min(abs(diag(B' * BOld2)));
                
                if (1 - minAbsCos < epsilon)
                    if finetuningEnabled & notFine
                        if b_verbose, fprintf('Initial convergence, fine-tuning: \n'); end;
                        notFine = 0;
                        usedNlinearity = gFine;
                        myy = myyK * myyOrig;
                        BOld = zeros(size(B));
                        BOld2 = zeros(size(B));
                        
                    else
                        if b_verbose, fprintf('Convergence after %d steps\n', round); end
                        
                        % Calculate the de-whitened vectors.
                        A = dewhiteningMatrix * B;
                        break;
                    end
                elseif stabilizationEnabled
                    if (~stroke) & (1 - minAbsCos2 < epsilon)
                        if b_verbose, fprintf('Stroke!\n'); end;
                        stroke = myy;
                        myy = .5*myy;
                        if mod(usedNlinearity,2) == 0
                            usedNlinearity = usedNlinearity + 1;
                        end
                    elseif stroke
                        myy = stroke;
                        stroke = 0;
                        if (myy == 1) & (mod(usedNlinearity,2) ~= 0)
                            usedNlinearity = usedNlinearity - 1;
                        end
                    elseif (~long) & (round>maxNumIterations/2)
                        if b_verbose, fprintf('Taking long (reducing step size)\n'); end;
                        long = 1;
                        myy = .5*myy;
                        if mod(usedNlinearity,2) == 0
                            usedNlinearity = usedNlinearity + 1;
                        end
                    end
                end
                
                BOld2 = BOld;
                BOld = B;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Show the progress...
                if b_verbose
                    if round == 1
                        fprintf('Step no. %d\n', round);
                    else
                        fprintf('Step no. %d, change in value of estimate: %.3g \n', round, 1 - minAbsCos);
                    end
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Also plot the current state...
                switch usedDisplay
                    case 1
                        if rem(round, displayInterval) == 0,
                            % There was and may still be other displaymodes...
                            % 1D signals
                            icaplot('dispsig',(mixedsig'*B)');
                            drawnow;
                        end
                    case 2
                        if rem(round, displayInterval) == 0,
                            % ... and now there are :-)
                            % 1D basis
                            A = dewhiteningMatrix * B;
                            icaplot('dispsig',A');
                            drawnow;
                        end
                    case 3
                        if rem(round, displayInterval) == 0,
                            % ... and now there are :-)
                            % 1D filters
                            W = B' * whiteningMatrix;
                            icaplot('dispsig',W);
                            drawnow;
                        end
                    otherwise
                end
                
                switch usedNlinearity
                    % pow3
                    case 10
                        B = (mixedsig * (( mixedsig' * B) .^ 3)) / numSamples - 3 * B;
                    case 11
                        % optimoitu - epsilonin kokoisia eroja
                        % t�m� on optimoitu koodi, katso vanha koodi esim.
                        % aikaisemmista versioista kuten 2.0 beta3
                        Y = mixedsig' * B;
                        Gpow3 = Y .^ 3;
                        Beta = sum(Y .* Gpow3);
                        D = diag(1 ./ (Beta - 3 * numSamples));
                        B = B + myy * B * (Y' * Gpow3 - diag(Beta)) * D;
                    case 12
                        Xsub=mixedsig(:, getSamples(numSamples, sampleSize));
                        B = (Xsub * (( Xsub' * B) .^ 3)) / size(Xsub,2) - 3 * B;
                    case 13
                        % Optimoitu
                        Ysub=mixedsig(:, getSamples(numSamples, sampleSize))' * B;
                        Gpow3 = Ysub .^ 3;
                        Beta = sum(Ysub .* Gpow3);
                        D = diag(1 ./ (Beta - 3 * size(Ysub', 2)));
                        B = B + myy * B * (Ysub' * Gpow3 - diag(Beta)) * D;
                        
                        % tanh
                    case 20
                        hypTan = tanh(a1 * mixedsig' * B);
                        B = mixedsig * hypTan / numSamples - ...
                            ones(size(B,1),1) * sum(1 - hypTan .^ 2) .* B / numSamples * ...
                            a1;
                    case 21
                        % optimoitu - epsilonin kokoisia
                        Y = mixedsig' * B;
                        hypTan = tanh(a1 * Y);
                        Beta = sum(Y .* hypTan);
                        D = diag(1 ./ (Beta - a1 * sum(1 - hypTan .^ 2)));
                        B = B + myy * B * (Y' * hypTan - diag(Beta)) * D;
                    case 22
                        Xsub=mixedsig(:, getSamples(numSamples, sampleSize));
                        hypTan = tanh(a1 * Xsub' * B);
                        B = Xsub * hypTan / size(Xsub, 2) - ...
                            ones(size(B,1),1) * sum(1 - hypTan .^ 2) .* B / size(Xsub, 2) * a1;
                    case 23
                        % Optimoitu
                        Y = mixedsig(:, getSamples(numSamples, sampleSize))' * B;
                        hypTan = tanh(a1 * Y);
                        Beta = sum(Y .* hypTan);
                        D = diag(1 ./ (Beta - a1 * sum(1 - hypTan .^ 2)));
                        B = B + myy * B * (Y' * hypTan - diag(Beta)) * D;
                        
                        % gauss
                    case 30
                        U = mixedsig' * B;
                        Usquared=U .^ 2;
                        ex = exp(-a2 * Usquared / 2);
                        gauss =  U .* ex;
                        dGauss = (1 - a2 * Usquared) .*ex;
                        B = mixedsig * gauss / numSamples - ...
                            ones(size(B,1),1) * sum(dGauss)...
                            .* B / numSamples ;
                    case 31
                        % optimoitu
                        Y = mixedsig' * B;
                        ex = exp(-a2 * (Y .^ 2) / 2);
                        gauss = Y .* ex;
                        Beta = sum(Y .* gauss);
                        D = diag(1 ./ (Beta - sum((1 - a2 * (Y .^ 2)) .* ex)));
                        B = B + myy * B * (Y' * gauss - diag(Beta)) * D;
                    case 32
                        Xsub=mixedsig(:, getSamples(numSamples, sampleSize));
                        U = Xsub' * B;
                        Usquared=U .^ 2;
                        ex = exp(-a2 * Usquared / 2);
                        gauss =  U .* ex;
                        dGauss = (1 - a2 * Usquared) .*ex;
                        B = Xsub * gauss / size(Xsub,2) - ...
                            ones(size(B,1),1) * sum(dGauss)...
                            .* B / size(Xsub,2) ;
                    case 33
                        % Optimoitu
                        Y = mixedsig(:, getSamples(numSamples, sampleSize))' * B;
                        ex = exp(-a2 * (Y .^ 2) / 2);
                        gauss = Y .* ex;
                        Beta = sum(Y .* gauss);
                        D = diag(1 ./ (Beta - sum((1 - a2 * (Y .^ 2)) .* ex)));
                        B = B + myy * B * (Y' * gauss - diag(Beta)) * D;
                        
                        % skew
                    case 40
                        B = (mixedsig * ((mixedsig' * B) .^ 2)) / numSamples;
                    case 41
                        % Optimoitu
                        Y = mixedsig' * B;
                        Gskew = Y .^ 2;
                        Beta = sum(Y .* Gskew);
                        D = diag(1 ./ (Beta));
                        B = B + myy * B * (Y' * Gskew - diag(Beta)) * D;
                    case 42
                        Xsub=mixedsig(:, getSamples(numSamples, sampleSize));
                        B = (Xsub * ((Xsub' * B) .^ 2)) / size(Xsub,2);
                    case 43
                        % Uusi optimoitu
                        Y = mixedsig(:, getSamples(numSamples, sampleSize))' * B;
                        Gskew = Y .^ 2;
                        Beta = sum(Y .* Gskew);
                        D = diag(1 ./ (Beta));
                        B = B + myy * B * (Y' * Gskew - diag(Beta)) * D;
                        
                    otherwise
                        error('Code for desired nonlinearity not found!');
                end
            end
            
            
            % Calculate ICA filters.
            W = B' * whiteningMatrix;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Also plot the last one...
            switch usedDisplay
                case 1
                    % There was and may still be other displaymodes...
                    % 1D signals
                    icaplot('dispsig',(mixedsig'*B)');
                    drawnow;
                case 2
                    % ... and now there are :-)
                    % 1D basis
                    icaplot('dispsig',A');
                    drawnow;
                case 3
                    % ... and now there are :-)
                    % 1D filters
                    icaplot('dispsig',W);
                    drawnow;
                otherwise
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DEFLATION APPROACH
        if approachMode == 2
            
            B = zeros(vectorSize);
            
            % The search for a basis vector is repeated numOfIC times.
            round = 1;
            
            numFailures = 0;
            
            while round <= numOfIC,
                myy = myyOrig;
                usedNlinearity = gOrig;
                stroke = 0;
                notFine = 1;
                long = 0;
                endFinetuning = 0;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Show the progress...
                if b_verbose, fprintf('IC %d ', round); end
                
                % Take a random initial vector of lenght 1 and orthogonalize it
                % with respect to the other vectors.
                if initialStateMode == 0
                    w = randn (vectorSize, 1);
                elseif initialStateMode == 1
                    w=whiteningMatrix*guess(:,round);
                end
                w = w - B * B' * w;
                w = w / norm(w);
                
                wOld = zeros(size(w));
                wOld2 = zeros(size(w));
                
                % This is the actual fixed-point iteration loop.
                %    for i = 1 : maxNumIterations + 1
                i = 1;
                gabba = 1;
                while i <= maxNumIterations + gabba
                    if (usedDisplay > 0)
                        drawnow;
                    end
                    if (interruptible & g_FastICA_interrupt)
                        if b_verbose
                            fprintf('\n\nCalculation interrupted by the user\n');
                        end
                        return;
                    end
                    
                    % Project the vector into the space orthogonal to the space
                    % spanned by the earlier found basis vectors. Note that we can do
                    % the projection with matrix B, since the zero entries do not
                    % contribute to the projection.
                    w = w - B * B' * w;
                    w = w / norm(w);
                    
                    if notFine
                        if i == maxNumIterations + 1
                            if b_verbose
                                fprintf('\nComponent number %d did not converge in %d iterations.\n', round, maxNumIterations);
                            end
                            round = round - 1;
                            numFailures = numFailures + 1;
                            if numFailures > failureLimit
                                if b_verbose
                                    fprintf('Too many failures to converge (%d). Giving up.\n', numFailures);
                                end
                                if round == 0
                                    A=[];
                                    W=[];
                                end
                                return;
                            end
                            % numFailures > failurelimit
                            break;
                        end
                        % i == maxNumIterations + 1
                    else
                        % if notFine
                        if i >= endFinetuning
                            wOld = w; % So the algorithm will stop on the next test...
                        end
                    end
                    % if notFine
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Show the progress...
                    if b_verbose, fprintf('.'); end;
                    
                    
                    % Test for termination condition. Note that the algorithm has
                    % converged if the direction of w and wOld is the same, this
                    % is why we test the two cases.
                    if norm(w - wOld) < epsilon | norm(w + wOld) < epsilon
                        if finetuningEnabled & notFine
                            if b_verbose, fprintf('Initial convergence, fine-tuning: '); end;
                            notFine = 0;
                            gabba = maxFinetune;
                            wOld = zeros(size(w));
                            wOld2 = zeros(size(w));
                            usedNlinearity = gFine;
                            myy = myyK * myyOrig;
                            
                            endFinetuning = maxFinetune + i;
                            
                        else
                            numFailures = 0;
                            % Save the vector
                            B(:, round) = w;
                            
                            % Calculate the de-whitened vector.
                            A(:,round) = dewhiteningMatrix * w;
                            % Calculate ICA filter.
                            W(round,:) = w' * whiteningMatrix;
                            
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Show the progress...
                            if b_verbose, fprintf('computed ( %d steps ) \n', i); end
                            
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Also plot the current state...
                            switch usedDisplay
                                case 1
                                    if rem(round, displayInterval) == 0,
                                        % There was and may still be other displaymodes...
                                        % 1D signals
                                        temp = mixedsig'*B;
                                        icaplot('dispsig',temp(:,1:numOfIC)');
                                        drawnow;
                                    end
                                case 2
                                    if rem(round, displayInterval) == 0,
                                        % ... and now there are :-)
                                        % 1D basis
                                        icaplot('dispsig',A');
                                        drawnow;
                                    end
                                case 3
                                    if rem(round, displayInterval) == 0,
                                        % ... and now there are :-)
                                        % 1D filters
                                        icaplot('dispsig',W);
                                        drawnow;
                                    end
                            end
                            % switch usedDisplay
                            break; % IC ready - next...
                        end
                        %if finetuningEnabled & notFine
                    elseif stabilizationEnabled
                        if (~stroke) & (norm(w - wOld2) < epsilon | norm(w + wOld2) < ...
                                epsilon)
                            stroke = myy;
                            if b_verbose, fprintf('Stroke!'); end;
                            myy = .5*myy;
                            if mod(usedNlinearity,2) == 0
                                usedNlinearity = usedNlinearity + 1;
                            end
                        elseif stroke
                            myy = stroke;
                            stroke = 0;
                            if (myy == 1) & (mod(usedNlinearity,2) ~= 0)
                                usedNlinearity = usedNlinearity - 1;
                            end
                        elseif (notFine) & (~long) & (i > maxNumIterations / 2)
                            if b_verbose, fprintf('Taking long (reducing step size) '); end;
                            long = 1;
                            myy = .5*myy;
                            if mod(usedNlinearity,2) == 0
                                usedNlinearity = usedNlinearity + 1;
                            end
                        end
                    end
                    
                    wOld2 = wOld;
                    wOld = w;
                    
                    switch usedNlinearity
                        % pow3
                        case 10
                            w = (mixedsig * ((mixedsig' * w) .^ 3)) / numSamples - 3 * w;
                        case 11
                            EXGpow3 = (mixedsig * ((mixedsig' * w) .^ 3)) / numSamples;
                            Beta = w' * EXGpow3;
                            w = w - myy * (EXGpow3 - Beta * w) / (3 - Beta);
                        case 12
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            w = (Xsub * ((Xsub' * w) .^ 3)) / size(Xsub, 2) - 3 * w;
                        case 13
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            EXGpow3 = (Xsub * ((Xsub' * w) .^ 3)) / size(Xsub, 2);
                            Beta = w' * EXGpow3;
                            w = w - myy * (EXGpow3 - Beta * w) / (3 - Beta);
                            % tanh
                        case 20
                            hypTan = tanh(a1 * mixedsig' * w);
                            w = (mixedsig * hypTan - a1 * sum(1 - hypTan .^ 2)' * w) / numSamples;
                        case 21
                            hypTan = tanh(a1 * mixedsig' * w);
                            Beta = w' * mixedsig * hypTan;
                            w = w - myy * ((mixedsig * hypTan - Beta * w) / ...
                                (a1 * sum((1-hypTan .^2)') - Beta));
                        case 22
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            hypTan = tanh(a1 * Xsub' * w);
                            w = (Xsub * hypTan - a1 * sum(1 - hypTan .^ 2)' * w) / size(Xsub, 2);
                        case 23
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            hypTan = tanh(a1 * Xsub' * w);
                            Beta = w' * Xsub * hypTan;
                            w = w - myy * ((Xsub * hypTan - Beta * w) / ...
                                (a1 * sum((1-hypTan .^2)') - Beta));
                            % gauss
                        case 30
                            % This has been split for performance reasons.
                            u = mixedsig' * w;
                            u2=u.^2;
                            ex=exp(-a2 * u2/2);
                            gauss =  u.*ex;
                            dGauss = (1 - a2 * u2) .*ex;
                            w = (mixedsig * gauss - sum(dGauss)' * w) / numSamples;
                        case 31
                            u = mixedsig' * w;
                            u2=u.^2;
                            ex=exp(-a2 * u2/2);
                            gauss =  u.*ex;
                            dGauss = (1 - a2 * u2) .*ex;
                            Beta = w' * mixedsig * gauss;
                            w = w - myy * ((mixedsig * gauss - Beta * w) / ...
                                (sum(dGauss)' - Beta));
                        case 32
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            u = Xsub' * w;
                            u2=u.^2;
                            ex=exp(-a2 * u2/2);
                            gauss =  u.*ex;
                            dGauss = (1 - a2 * u2) .*ex;
                            w = (Xsub * gauss - sum(dGauss)' * w) / size(Xsub, 2);
                        case 33
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            u = Xsub' * w;
                            u2=u.^2;
                            ex=exp(-a2 * u2/2);
                            gauss =  u.*ex;
                            dGauss = (1 - a2 * u2) .*ex;
                            Beta = w' * Xsub * gauss;
                            w = w - myy * ((Xsub * gauss - Beta * w) / ...
                                (sum(dGauss)' - Beta));
                            % skew
                        case 40
                            w = (mixedsig * ((mixedsig' * w) .^ 2)) / numSamples;
                        case 41
                            EXGskew = (mixedsig * ((mixedsig' * w) .^ 2)) / numSamples;
                            Beta = w' * EXGskew;
                            w = w - myy * (EXGskew - Beta*w)/(-Beta);
                        case 42
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            w = (Xsub * ((Xsub' * w) .^ 2)) / size(Xsub, 2);
                        case 43
                            Xsub=mixedsig(:,getSamples(numSamples, sampleSize));
                            EXGskew = (Xsub * ((Xsub' * w) .^ 2)) / size(Xsub, 2);
                            Beta = w' * EXGskew;
                            w = w - myy * (EXGskew - Beta*w)/(-Beta);
                            
                        otherwise
                            error('Code for desired nonlinearity not found!');
                    end
                    
                    % Normalize the new w.
                    w = w / norm(w);
                    i = i + 1;
                end
                round = round + 1;
            end
            if b_verbose, fprintf('Done.\n'); end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Also plot the ones that may not have been plotted.
            if (usedDisplay > 0) & (rem(round-1, displayInterval) ~= 0)
                switch usedDisplay
                    case 1
                        % There was and may still be other displaymodes...
                        % 1D signals
                        temp = mixedsig'*B;
                        icaplot('dispsig',temp(:,1:numOfIC)');
                        drawnow;
                    case 2
                        % ... and now there are :-)
                        % 1D basis
                        icaplot('dispsig',A');
                        drawnow;
                    case 3
                        % ... and now there are :-)
                        % 1D filters
                        icaplot('dispsig',W);
                        drawnow;
                    otherwise
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % In the end let's check the data for some security
        if ~isreal(A)
            if b_verbose, fprintf('Warning: removing the imaginary part from the result.\n'); end
            A = real(A);
            W = real(W);
        end
    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction
% Calculates tanh simplier and faster than Matlab tanh.
    function y=tanh(x)
        y = 1 - 2 ./ (exp(2 * x) + 1);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function Samples = getSamples(max, percentage)
        Samples = find(rand(1, max) < percentage);
    end


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END FastICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = dircat(subj, ictyp)

[fnames, mydir] = subjtyp2dirs(subj, ictyp, 'raw');
nFle = length(fnames);
x = cell(nFle,1);
for iFle=1:nFle;
    x{iFle} = fullfile(mydir,fnames{iFle});
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wfnamefull,Wfnamefull_log] = getWfname(subj)

% Declarations
settingsfname = 'SETTINGS.json';

% Load the settings file
settings = json.read(settingsfname);

mydir = fullfile(getRepoDir(), settings.MODEL_PATH);
Wfname = ['ica_weights_' subj];

Wfnamefull = fullfile(mydir,Wfname);
Wfnamefull_log = fullfile(mydir,'log',[Wfname '_' datestr(now,30)]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function icafname = raw2icafname(fname)

% Decompose existing file
[pth,fname,ext] = fileparts(fname);

% Go up a directory, so we are before the subject name
[pth,subj] = fileparts(pth);

% Add ica folder between data directory and subject name
pth = fullfile(pth,'ica',subj);

% Add ica label in front of file name
fname = ['ica_' fname];

% Recombine new path, filename and extension
icafname = fullfile(pth,[fname ext]);

end
