
subprob1 = {'Dog_1','Dog_2'};

allsub = subjnames();

subprob1str = sprintf('-%s',subprob1{:});
outnme = sprintf('testsubmission%s-%s.csv',subprob1str,datestr(now,30));
outdir = fullfile(getRepoDir(), 'output');

outfile = fullfile(outdir,outnme);

if exist(outfile,'file');
    error('Output file already exists');
end

fid = fopen(outfile,'w+');

fprintf(fid,'clip,preictal');

for iSub=1:length(allsub)
    if ismember(allsub{iSub},subprob1)
        p = 1;
    else
        p = 0;
    end
    
    [fnames] = subjtyp2dirs(allsub{iSub}, 'test', 'raw');
    
    fprintf(fid, ['\n%s,' num2str(p)], fnames{:});
    
end

fclose(fid);