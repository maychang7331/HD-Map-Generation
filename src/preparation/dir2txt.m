function outfilename = dir2txt(filepath, filetype)
d = dir(fullfile(filepath, filetype));
fprintf(filetype);
% 建立meta資料夾
currentDir = convertCharsToStrings(pwd)
metaDir = currentDir + '/meta';     % = output directory
if ~exist(metaDir, 'dir')
    mkdir(metaDir);
end

% 將las寫入txt
parts = strsplit(filepath,'/');
outfilename = metaDir + '/' + parts{end-1} + '_' + parts{end}  + '_' + 'dir.txt' 
fid = fopen(outfilename,'w');
for i = 1:length(d)
    fprintf(fid, '%s/%s\n',d(i).folder,d(i).name);
end
fclose(fid);
end