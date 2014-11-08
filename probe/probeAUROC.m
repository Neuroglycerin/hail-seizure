
AUROC_Dog_1    = 0.49128;
AUROC_Dog_2    = 0.55855;
AUROC_Dog_3    = 0.46885;
AUROC_Dog_4    = 0.47984;
AUROC_Dog_5    = 0.49809;
AUROC_Patient_1= 0.48837;
AUROC_Patient_2= 0.51501;

accuracy_AUROC = 0.000005;

nTest_Dog_1     = 502;
nTest_Dog_2     =1000;
nTest_Dog_3     = 907;
nTest_Dog_4     = 990;
nTest_Dog_5     = 191;
nTest_Patient_1 = 195;
nTest_Patient_2 = 150;

AUROC = [AUROC_Dog_1; AUROC_Dog_2; AUROC_Dog_3; AUROC_Dog_4; AUROC_Dog_5; AUROC_Patient_1; AUROC_Patient_2];
nTest = [nTest_Dog_1; nTest_Dog_2; nTest_Dog_3; nTest_Dog_4; nTest_Dog_5; nTest_Patient_1; nTest_Patient_2];

dbAm1 = 2*AUROC - 1;
accuracy_dbAm1 = 2*accuracy_AUROC;
nSubj = length(AUROC);

% Identities, true for all k:
% nPos_k / Sum(nPos) - nNeg_k / Sum(nNeg) = 2 * (AUROC_k - 0.5)
% which is to say
% nPos_k / Sum(nPos) - nNeg_k / Sum(nNeg) = 2 * AUROC_k - 1
% nPos_k + nNeg_k = nTestUse_k
% nTestUse_k = 0.4(+/-0.05) * nTest_k
% nPos_k and nNeg_k are both integers
% nPos_k / nTestUse_k is around 0.074 (7.5%)

% Metric to use
% best_distance = sum(max(0, abs(dbAm1 - (nPos / sum(nPos) - (nTestUse-nPos) / sum(nTestUse-nPos))) - accuracy_AUROC));
testfunc = @(x)( max(max(0, abs(dbAm1' - (x(1:nSubj)/sum(x(1:nSubj)) - (x(nSubj+1:end)-x(1:nSubj))/sum(x(nSubj+1:end)-x(1:nSubj)))) - accuracy_dbAm1)) );
aurocguess = @(x)(0.5 + 0.5 * (x(1:nSubj)/sum(x(1:nSubj)) - (x(nSubj+1:end)-x(1:nSubj))/sum(x(nSubj+1:end)-x(1:nSubj))));

% Initialise
nTestUse = round(nTest * 0.4);
nPos = round(nTestUse * 0.074);
% x0 = [nPos; nTestUse];
x0 = [9.6; 31.0; 17.3; 42.6; 4.8; 20.6; 18.0; nTestUse];
x0 = [12; 36; 18; 36; 5; 18; 18; nTestUse];
x0 = [12; 36; 18; 30; 6; 6; 6; nTestUse];
expected_prop = [0.0478 0.0775 0.0477 0.1076 0.0632 0.2641 0.3000];
x1 = [12    37    16    26     6     5     5   215   414   367   486    97    80    75];

x2 = [18    55    29    31     7     5    11   202   391   374   358    80    87    64];

return;


%% Simulated annealing. Cannot restrict to integers
% options = saoptimset(@simulannealbnd);
% options.InitialTemperature = 100000000000;
% options.ReannealInterval = 100000000000;
% [x,fval] = simulannealbnd(testfunc,x0,lb,ub);


%%

% Constraints
% lb = [3*ones(nSubj,1); floor(nTest*0.25)];
% ub = ceil([nTest*0.3; nTest*0.6]);
% lb = [3*ones(nSubj,1); floor(nTest*0.34)];
% ub = ceil([nTest*0.3; nTest*0.46]);
lb = [3*ones(nSubj,1); nTest*0.4 - 12 - nTest*0.02];
ub = ceil([nTest*0.3; nTest*0.4 + 12 + nTest*0.02]);

%                0.0597 0.0900 0.0496 0.0758 0.0789 0.0769 0.1000
expected_prop = [0.0478 0.0775 0.0477 0.1076 0.0632 0.2641 0.3000];
A = []; b = [];
A(end+1,1:14) = [1 0 0 0 0 0 0 -0.10 0 0 0 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [-1 0 0 0 0 0 0 0.03 0 0 0 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 1 0 0 0 0 0 0 -0.15 0 0 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 -1 0 0 0 0 0 0 0.05 0 0 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 1 0 0 0 0 0 0 -0.10 0 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 -1 0 0 0 0 0 0 0.03 0 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 1 0 0 0 0 0 0 -0.15 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 -1 0 0 0 0 0 0 0.05 0 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 1 0 0 0 0 0 0 -0.15 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 -1 0 0 0 0 0 0 0.04 0 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 0 1 0 0 0 0 0 0 -0.30 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 0 -1 0 0 0 0 0 0 0.04 0]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 0 0 1 0 0 0 0 0 0 -0.30]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 0 0 -1 0 0 0 0 0 0 0.04]; b(end+1,1) = 0;
A(end+1,1:14) = [0 0 0 0 0 0 0  1 1 1 1 1 1 1   ]; b(end+1,1) = sum(nTest)*0.45;
A(end+1,1:14) = [0 0 0 0 0 0 0 -1 -1 -1 -1 -1 -1 -1]; b(end+1,1) = -sum(nTest)*0.35;

% Genetic algorithm. Can restict to integers.
options = gaoptimset(@ga);
options.PopulationSize = 10000;
x = ga(testfunc,2*nSubj,[],[],[],[],lb',ub',[],1:2*nSubj,options);

%%
convergentmethod = false;
if convergentmethod
    prb = expected_prop(:);
    tk = round(nTest(:) * 0.4);
    pk = prb.*tk;
    nk = tk-pk;
    P = sum(pk);
    N = sum(nk);
    A = AUROC(:);
    pk_last = pk;
    reachedthrlim = false;
    fprintf('pk_initial = ['); fprintf('%.3f ',pk); fprintf(']\n');
    for i=1:1000
        tk = round(tk + randi(5,size(tk))-3);
        pk = max(5, round(pk + randi(3,size(tk))-2));
        P = sum(pk);
        N = sum(tk-pk);
        % pk = P*tk+2*(A-0.5)*N*P/(N+P);
        pk = P/(N+P)*(tk+2*(A-0.5)*N);
        fprintf('pk = ['); fprintf('%.4f ',pk); fprintf(']\n');
        if max(abs(pk_last-pk))<1e-3; reachedthrlim=true; break; end;
        pk_last = pk;
    end
    if reachedthrlim;
        fprintf('Reached threshold\n');
    else
        fprintf('Reached count\n');
    end
    return;
end

%%
bruteForce = false;
if bruteForce
    besty = 100000000;
    for iT1=2:nTest_Dog_1*0.5
    for iT2=2:nTest_Dog_2*0.5
    for iT3=2:nTest_Dog_3*0.5
    for iT4=2:nTest_Dog_4*0.5
    for iT5=2:nTest_Dog_5*0.5
    for iT6=2:nTest_Patient_1*0.5
    for iT7=2:nTest_Patient_2*0.5
    for iP1=1:iT1*0.5
    for iP2=1:iT2*0.5
    for iP3=1:iT3*0.5
    for iP4=1:iT4*0.5
    for iP5=1:iT5*0.5
    for iP6=1:iT6*0.5
    for iP7=1:iT7*0.5
        x = [iP1;iP2;iP3;iP4;iP5;iP6;iP7;iT1;iT2;iT3;iT4;iT5;iT6;iT7];
        y = testfunc(x);
        if y<=besty
            fprintf('New best: %.6f\n',y);
            disp(x');
            besty = y;
            bestx = x;
        end
    end
    end
    end
    end
    end
    end
    end
    end
    end
    end
    end
    end
    end
    end

    return;
end
