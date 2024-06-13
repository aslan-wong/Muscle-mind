clear;
close all;
%% Import data
Sheet = ["Trial left" "Trial right" "MVC left" "MVC right"];
for i = 1:4
    Data{i} = readtable("Hamstring curl.xlsx","Sheet",Sheet(i));
end


%% Save data into a cell array
EMG{1} = Data{1}(:,4:5);
EMG{2} = Data{2}(:,2:3);
EMG{3} = Data{3}(:,4:5);
EMG{4} = Data{4}(:,2:3);


%% Plot raw data
Muscles = ["Bicep femoris" "Semi-tendinosus"];
for i = 1:4
    fig = figure;
    for k = 1:2
        Mat = table2array(EMG{i}(:,k));
        sub(k) = subplot(2,1,k);
        plot((0:length(Mat)-1)*(1/2000),Mat);
        title(Muscles(k));
    end
    xlabel("Time [s]");
    subtitle(Sheet(i));
    P1 = get(sub(1), 'position');
    P2 = get(sub(2), 'position');
    height = P1(2) + P1(4) - P2(2);
    AX = axes('position',[P2(1) P2(2) P2(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
    
    saveas(fig,strcat(Sheet(i),".png"));
    fig = [];
end


%% Power spectrum
for i = 1:4
    figure;
    for k = 1:2
        Mat = table2array(EMG{i}(:,k));
        [P,f] = pspectrum(Mat,2000);
        sub(k) = subplot(2,1,k);
        plot(f,pow2db(P));
        title(Muscles(k));
    end
    xlabel("Frequency");
    subtitle(Sheet(i));
    P1 = get(sub(1), 'position');
    P2 = get(sub(2), 'position');
    height = P1(2) + P1(4) - P2(2);
    AX = axes('position',[P2(1) P2(2) P2(3) height], 'visible','off'); 
    ylabel('Power spectrum [dB]','visible','on');
end


%% Detrend data
for i = 1:4
    Detrend_Data{i} = detrend(table2array(EMG{i}));
end


%% Filter data
fs = 2000;
[b, a] = butter(2,[10/(fs/2) 500/(fs/2)],'bandpass');
for i = 1:4
    for k = 1:2
        Mat = Detrend_Data{i}(:,k);
        Filter(:,k) = filtfilt(b,a,Mat);
    end
    Filtered_data{i} = Filter;
    Filter = [];
    Mat = [];
end

% Plot the filtered data
for i = 1:4
    figure,
    for k = 1:2
        sub(k) = subplot(2,1,k);
        plot((0:length(Filtered_data{i}(:,k))-1)*(1/2000),Filtered_data{i}(:,k));
        title(Muscles(k));
        ylim([-1 1]);
    end
    xlabel("Time [s]");
    subtitle(Sheet(i));
    P1 = get(sub(1), 'position');
    P2 = get(sub(2), 'position');
    height = P1(2) + P1(4) - P2(2);
    AX = axes('position',[P2(1) P2(2) P2(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
end


%% Rectify filtered data
for i = 1:4
    Rec_filtered_data{i} = abs(Filtered_data{i});
end


%% Apply moveing average
for i = 1:4
    Movav_data{i} = movmean(Rec_filtered_data{i},150,1);
end

% Plot the data
for i = 1:4
    figure,
    for k = 1:2
        sub(k) = subplot(2,1,k);
        plot((0:length(Movav_data{i}(:,k))-1)*(1/2000),Movav_data{i}(:,k));
        title(Muscles(k));
        ylim([0 0.3]);
    end
    xlabel("Time [s]");
    subtitle(Sheet(i));
    P1 = get(sub(1), 'position');
    P2 = get(sub(2), 'position');
    height = P1(2) + P1(4) - P2(2);
    AX = axes('position',[P2(1) P2(2) P2(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
end


%% Find peak MVC
for i = 1:2
    Peak(i,1:2) = max(Movav_data{i+2});
end


%% Normalize data
for i = 1:2
    for k = 1:2
        N(:,k) = Movav_data{i}(:,k)/Peak(i,k)*100; 
    end
    Norm_data{i} = N;
    N = [];
end

% Plot the data
L = ["L Bicep" "L Semi" "R Bicep" "R Semi"];

fig = figure;
sub(1) = subplot(2,1,1);
for i = 1:2
    for k = 1:2
        plot((0:length(Norm_data{i}(:,k))-1)*(1/2000),Norm_data{i}(:,k)),hold on;
    end
end
legend(L,"Location","bestoutside");
title("Normalization and envelope");
ylabel("MVC [%]");
sub(2) = subplot(2,1,2);
for i = 1:2
    for k = 1:2
        plot((0:length(Movav_data{i}(:,k))-1)*(1/2000),Movav_data{i}(:,k)),hold on;
    end
    xlabel("Time [s]");
    ylabel('Amplitude [V]','visible','on');
end
legend(L,"Location","bestoutside");

saveas(fig,"Normalized and envelope.png");




