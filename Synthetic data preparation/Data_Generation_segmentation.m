clear
clc

current_path = pwd;
cd E:\human-protein-atlas-image-classification\train
saved_path = 'E:\training_seg\';

if ~exist(saved_path, 'dir')
    mkdir(saved_path)
end
a = dir('*.png');

N = 1000;
RGB_channel =1;
tic

parfor i = 3001:N+3001
    if rand()<0.7
        RGB_channel = 1;
    else
        RGB_channel = 4;
    end
    GenSynFLI_seg(i,RGB_channel,saved_path)
    disp(['Finished',num2str(i),'/',num2str(N)])
end

toc
cd (current_path)