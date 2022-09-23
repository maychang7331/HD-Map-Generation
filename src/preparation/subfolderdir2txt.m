function subfolderdir2txt(in_folderdir)
d = dir(in_folderdir);
subfolders = d([d(:).isdir]);     % remove all files (isdir property is 0)
subfolders = subfolders(~ismember({subfolders(:).name}, {'.', '..'}));	% remove '.' and '..'

parentFolder = fileparts(in_folderdir)
parts = strsplit(in_folderdir, '/');
parts{end};
outdir = parentFolder + '/' + parts{end} + '_path.txt'
fid = fopen(outdir,'w');
for i = 1:length(subfolders)
    fprintf(fid,'%s/%s\n',subfolders(i).folder,subfolders(i).name);
end
fclose(fid);
end