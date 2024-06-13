function fft_data = calFFT(X, fs)
    N = length(X);
%     figure;plot(X);title('raw signal');
    [L,a] = size(X);
%     figure;plot(X);title('autocorrelation ');
    Y = fft(X,N);
    P2 = abs(Y/L);% tow-side spectrum
    P1 = P2(1:floor(N/2)+1)';%One-sided
    f = fs*(0:(N/2))/N;%sample rate is Fs, only need the signal in FS /2
    fft_data=[f;P1];
    figure;plot(f, P1);
end