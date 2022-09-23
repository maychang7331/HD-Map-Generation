clc;
clear all;

% 要選change folder才會在thesis下的meta資料夾
folder = '/home/chihyu/Desktop/thesis/data/shalun_octree/8';
lasfileList = dir(fullfile(folder, '*.las'));
for i = 1:length(lasfileList)
    lasfile = fullfile(lasfileList(i).folder, lasfileList(i).name)
    las = readlas(lasfile);
    [filepath,name,ext] = fileparts(lasfile); 
    parts = strsplit(filepath,'/');
    currentDir = convertCharsToStrings(pwd);                % "/home/chihyu/Desktop/thesis"
    outDir = currentDir + '/data/shalun_txt/' + parts{end};	% = output directory
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    writetable(struct2table(las), [outDir '/' name '.txt'],'WriteVariableNames', false,'Delimiter',' ');
end
dir2txt(outDir, '*.txt');
fprintf("finish converting las to txt !");