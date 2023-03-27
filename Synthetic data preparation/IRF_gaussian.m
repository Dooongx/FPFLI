function I = IRF_gaussian(t0, h,FWHM)
% Generate synthetic  IRF
% t0:  start bin
% FWHM: Full width half maximum, FWHM = 0.1673/h; 

t=1:256;   %observation window,suppose 10 time bin width, h=0.04ns
%IRF  suppose Gaussian function, FWHM=2.35*sig0
sig0 =FWHM/2.3548/h;
I=exp(-(t-t0).^2/(2*sig0^2));

end