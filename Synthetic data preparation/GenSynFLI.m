function GenSynFLI(ind,tau,RGB_channel,path)

assert (length(tau)==length(RGB_channel),'The length of tau and RGB_channel should be equal')
N = length(RGB_channel);
Size = 256; % image size
bin_Num = 256; % bin number
thres = 20; % lower intensity threshold 
h = 0.039;
IRF = IRF_gaussian(14, h,0.1673);

a = dir('*.png');
[x, xo, x_1d, tau_map, tau_map_1d, hist_1d, hist]= deal({N});
 

for  i = 1:N
    
    x{i} = imread(a(ind*4+RGB_channel(i)).name);
    x{i} = double(imresize(x{i},[Size Size],'bicubic'));
    xo{i} = x{i}; % original intensity 
    M = mean(x{i}(x{i}>thres),'all');
    Ave = randi(4)+6;
    x{i} = round(x{i}/(M+1e-6)*Ave); % GT intensity
    x_1d{i} = x{i}(:);
    tau_map_1d{i} =  zeros(Size *Size ,1);
    hist_1d{i} =  zeros(Size *Size ,bin_Num);
    
end

% generate temporal decays for each pixel
Hist = zeros([Size, Size, bin_Num]);
for i = 1:N
    for j = 1:Size*Size
        if (x_1d{i}(j)>0)
            hist_1d{i}(j,:) = Fluorescence_multi_decay_nonhomopp(1,tau(i),1,poissrnd(x_1d{i}(j)),h,IRF);
            tau_map_1d{i}(j) = tau(i);
        else
            hist_1d{i}(j,:) = zeros(1,bin_Num);
        end
    end
    hist{i} = reshape(hist_1d{i},[Size, Size, bin_Num]);
    tau_map{i} = reshape(tau_map_1d{i},[Size, Size]);
    Hist = Hist + hist{i};
end

% Obtain GT tau
[int, tau_gt] = deal(zeros([Size, Size]));
for i = 1:N
    int = int +x{i};
end

for i = 1:N
    tau_gt = tau_gt+ x{i}./(int+1e-8).*tau_map{i};
end

% output data
name =['Sample_', num2str(ind),'_'];
for i = 1:N
    switch RGB_channel(i)
    case 1
        suffix = 'C1';
    case 2
        suffix = 'C2';
    case 3
        suffix = 'C3';
    case 4
        suffix = 'C4';
    end
   name = append(name,suffix);
end
save([path,name],'Hist','tau_gt')

end

