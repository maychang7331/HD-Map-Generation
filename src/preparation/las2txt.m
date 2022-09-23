clc;
clear all;

% 要選change folder才會在thesis下的meta資料夾
folder = '/home/chihyu/Desktop/shalun/testing_ts';
fout = '/home/chihyu/Desktop/shalun/testing_ts';
if ~exist(fout, 'dir')
    mkdir(fout);
end

lasfileList = dir(fullfile(folder, '*.las'));
for j = 1:length(lasfileList)
        lasfile = fullfile(lasfileList(j).folder, lasfileList(j).name)
        [filepath,name,ext] = fileparts(lasfile);
        las = readlas(lasfile);
        writetable(struct2table(las), [fout '/' name '.txt'],'WriteVariableNames', false,'Delimiter',' ')
end