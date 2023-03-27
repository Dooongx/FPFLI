clear
clc

current_path = pwd;
cd E:\human-protein-atlas-image-classification\train % HPA datasets
saved_path = 'E:\training\'; % saved path

if ~exist(saved_path, 'dir')
    mkdir(saved_path)
end

a = dir('*.png');

N = 5000;

tic
parfor i = 1:N
    if rand()<0.2
        RGB_channel = [1,2,4];
        tau = 1+rand(1,3)*3;
    elseif rand()<0.6 && rand()>=0.2
        RGB_channel = [1,4];
        tau = 1+rand(1,2)*3;
    else
        RGB_channel = [2,3];
        tau = 1+rand(1,2)*3;
    end
    GenSynFLI(i,tau,RGB_channel,saved_path)
    disp(['Finished: ',num2str(i),'/',num2str(N)])
end
toc
cd (current_path)