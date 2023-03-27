function GenSynFLI_seg(ind,RGB_channel,path)

a = dir('*.png');
Size=256;
bin_Num = 256;
thres = 20; 
h = 0.039;
IRF = IRF_gaussian(14, h,0.1673);

x1 = imread(a(ind*4+RGB_channel(1)).name);
x1 = imresize(x1,[Size Size],'bicubic');
x1 = double(x1);
M = mean(x1(x1>thres),'all');
Ave = randi(4)+6;
x1 = round(x1./(M+1e-6)*Ave); % GT intensity

if any(isnan(x1))
    x1(isnan(x1)) = 0;
end
s1_b = logical(x1); % binary image
[label_1, N_1 ]= bwlabel(s1_b);
N_tau = randi(3); % lifetime components

for i = 1:N_1
    label_1(label_1 == i) = randi(N_tau);
end

%% add the lifetime dimension
%Assign lifetime
tau = rand(1,N_tau)*3+1;
x1_1d = x1(:);
% channel_Num = 256;
tau_map_1d = zeros(Size*Size,1);
decay_c1 = zeros(Size*Size,bin_Num);

for i = 1:Size*Size
    if (x1_1d(i)>1) 
        decay_c1(i,:) = Fluorescence_multi_decay_nonhomopp(1,tau(i),1,poissrnd(x1_1d(i)),h,IRF);
        tau_map_1d(i) = tau(label_1(i));
    end
end

Hist = reshape(decay_c1,[Size,Size,bin_Num]);
tau_gt = reshape(tau_map_1d,[Size,Size]);
save([path,'Sample_',num2str(ind),'_',num2str(RGB_channel),'_seg'],'Hist','tau_gt')


end