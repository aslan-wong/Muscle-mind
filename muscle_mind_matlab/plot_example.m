close all;

load('data_example/1.mat');
data1 = example;
load('data_example/2.mat');
data2 = example;
load('data_example/5.mat');
data3 = example;
load('data_example/6.mat');
data4 = example;
load('data_example/13.mat');
data5 = example;
fs = 200;
[b,a] = butter(2,1/fs*2,'high');
data1 = smooth(data1);
data2 = smooth(data2);
data3 = smooth(data3);
data4 = smooth(data4);
data5 = smooth(data5);
% data1 = (filter(b,a, data1));
% data2 = (filter(b,a, data2));
% data3 = (filter(b,a, data3));
% data4 = (filter(b,a, data4));
% % data5 = (filter(b,a, data5));
% data1=data1(50:end);
% data2=data2(50:end);
% data3=data3(50:end);
% data4=data4(50:end);
% data5=data5(50:end);
% plot(data1);hold on;
plot(data2);hold on;
plot(data3);hold on;
plot(data4);hold on;
plot(data5);
figure;plot(xcorr(data1,data2,'unbiased'))
figure;plot(xcorr(data1,data3,'unbiased'))
figure;plot(xcorr(data1,data4,'unbiased'))
figure;plot(xcorr(data1,data5,'unbiased'))

[~,a2] = max(abs(xcorr(data1,data2,'unbiased')));
[~,a3] = max(abs(xcorr(data2,data3,'unbiased')));
[~,a4] = max(abs(xcorr(data2,data4,'unbiased')));
[~,a5] = max(abs(xcorr(data2,data5,'unbiased')));

figure;plot(data1);hold on;
plot(data2(501-a2:end));hold on;
% plot(data3(501-a3:end));hold on;
plot(data4(501-a4:end));hold on;
plot(data5(501-a5:end));