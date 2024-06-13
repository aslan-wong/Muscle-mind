close all;
load('train_data.mat');
load('test_data.mat');
load('val_data.mat');
fs = 200;

train_n = size(train_data, 3);
test_n = size(test_data, 3);
val_n = size(val_data, 3);
train_data_2d = zeros(65,74,8,train_n);
test_data_2d = zeros(65,74,8,test_n);
val_data_2d = zeros(65,74,8,val_n);
train_data_1d = zeros(74,8,train_n);
test_data_1d = zeros(74,8,test_n);
for i = 1 : train_n
    for j = 1 : 8
        raw_data = squeeze(train_data(:,j,i));
        stft = get_stft(raw_data, fs);
        train_data_2d(:,:,j,i) = stft;
%         train_data_1d(:,j,i) = squeeze(mean(stft));
    end
end
for i = 1 : test_n
    for j = 1 : 8
        raw_data = squeeze(test_data(:,j,i));
        stft = get_stft(raw_data, fs);
        test_data_2d(:,:,j,i) = stft;
%         test_data_1d(:,j,i) = squeeze(mean(stft));
    end
end
for i = 1 : val_n
    for j = 1 : 8
        raw_data = squeeze(val_data(:,j,i));
        stft = get_stft(raw_data, fs);
        val_data_2d(:,:,j,i) = stft;
%         test_data_1d(:,j,i) = squeeze(mean(stft));
    end
end
save('train_data_2d.mat','train_data_2d')
save('val_data_2d.mat','val_data_2d')
% save('train_data_1d.mat','train_data_1d')
save('test_data_2d.mat','test_data_2d')
% save('test_data_1d.mat','test_data_1d')
function feature = get_stft(data, fs)
    win = 0.3;
    [b,a] = butter(2,1/fs*2,'high');
    raw_data = filter(b,a, data);
%     calFFT(raw_data(1:200,:), fs);
    [S, F, T, P] = spectrogram(data,win*fs,floor(win*0.9*fs),2^nextpow2(win*fs)*2,fs,'yaxis');  
%     figure;imagesc(T,F,10*log10(P));
    feature = 10*log10(P);
    max_value = max(max(feature));
    min_value = min(min(feature));
%     feature = (feature - min_value) / (max_value - min_value);
end
