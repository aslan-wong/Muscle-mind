%% Import data
Sheet = ["Nordic 1" "Nordic 2" "Nordic 3" "Nordic 4" "Nordic 5"];
for i = 1:5
    Data{i} = readtable("EMG Nordic.xlsx","Sheet",Sheet(i));
end


%% Plot raw data
Muscles = ["R Biceps" "R Semi" "L Biceps" "L Semi"];
for i = 1:5
    Name = strcat(Muscles," [V]");
    figure;
    stack = stackedplot(Data{i}(:,2:5),'Title',Sheet(i),'DisplayLabels',Name(1:4));
    stack.AxesProperties(1).YLimits = [-2 2]; % Lose tick labels !!
    stack.AxesProperties(2).YLimits = [-2 2];
    stack.AxesProperties(3).YLimits = [-2 2];
    stack.AxesProperties(4).YLimits = [-2 2];
    xlabel("Time [s]");
end


%% Detrend raw data
for i = 1:5
    Detrend_Data{i} = detrend(table2array(Data{i}(:,2:5)));
end

% Plot the envelope
for i = 1:5
    figure,
    for k = 1:4
        sub(k) = subplot(4,1,k);
        plot(Detrend_Data{i}(:,k));
        title(Muscles(k));
        ylim([-2 2]);
    end
    xlabel("Time [s]");
    suptitle(Sheet(i));
    P2 = get(sub(2), 'position');
    P3 = get(sub(3), 'position');
    height = P2(2) + P2(4) - P3(2);
    AX = axes('position',[P3(1) P3(2) P3(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
end


%% Power spectrum
for i = 1:5
    fig4 = figure;
    for k = 1:4
        Mat = Detrend_Data{i}(:,k);
        [P,f] = pspectrum(Mat,2000);
        sub(k) = subplot(4,1,k);
        plot(f,pow2db(P));
        title(Muscles(k));
    end
    xlabel("Frequency");
    suptitle(Sheet(i));
    P2 = get(sub(2), 'position');
    P3 = get(sub(3), 'position');
    height = P2(2) + P2(4) - P3(2);
    AX = axes('position',[P3(1) P3(2) P3(3) height], 'visible','off'); 
    ylabel('Power spectrum [dB]','visible','on');
end

saveas(fig4,"Power spectrum R Biceps.png")

%% Filter the data
fs = 2000;
[b, a] = butter(2,[10/(fs/2) 500/(fs/2)],'bandpass');
for i = 1:5
    for k = 1:4
        Mat = Detrend_Data{i}(:,k);
        Filter(:,k) = filtfilt(b,a,Mat);
    end
    Filtered_data{i} = Filter;
    Filter = [];
    Mat = [];
end

% Plot the filtered data
for i = 1:5
    figure,
    for k = 1:4
        sub(k) = subplot(4,1,k);
        plot(Filtered_data{i}(:,k));
        title(Muscles(k));
        ylim([-2 2]);
    end
    xlabel("Time [s]");
    suptitle(Sheet(i));
    P2 = get(sub(2), 'position');
    P3 = get(sub(3), 'position');
    height = P2(2) + P2(4) - P3(2);
    AX = axes('position',[P3(1) P3(2) P3(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
end


%% Rectify filtered data
for i = 1:5
    Rec_filtered_data{i} = abs(Filtered_data{i});
end

% Plot the rectified data
for i = 1:5
    figure,
    for k = 1:4
        sub(k) = subplot(4,1,k);
        plot(Rec_filtered_data{i}(:,k));
        title(Muscles(k));
        ylim([-2 2]);
    end
    xlabel("Time [s]");
    suptitle(Sheet(i));
    P2 = get(sub(2), 'position');
    P3 = get(sub(3), 'position');
    height = P2(2) + P2(4) - P3(2);
    AX = axes('position',[P3(1) P3(2) P3(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
end


%% Apply moveing average
for i = 1:5
    Movav_data{i} = movmean(Rec_filtered_data{i},2000,1);
end

% Plot the data
for i = 1:5
    figure,
    for k = 1:4
        sub(k) = subplot(4,1,k);
        plot(Movav_data{i}(:,k));
        title(Muscles(k));
        ylim([0 0.6]);
    end
    xlabel("Time [s]");
    suptitle(Sheet(i));
    P2 = get(sub(2), 'position');
    P3 = get(sub(3), 'position');
    height = P2(2) + P2(4) - P3(2);
    AX = axes('position',[P3(1) P3(2) P3(3) height], 'visible','off'); 
    ylabel('Amplitude [V]','visible','on');
end


%% Save figure for presentation
fig1 = figure;
plot(Data{1}.R_Biceps);
ylabel("Amplitude [V]");
xlabel("Time [s]");
title("EMG on right bicep femoris");

saveas(fig1,"Raw EMG R Biceps.png");


fig2 = figure;
plot(Movav_data{1}(:,1));
xlabel("Time [s]");
ylabel("Amplitude [V]");
title("EMG on right bicep femoris");

saveas(fig2,"Enveloped EMG R Biceps.png");


fig3 = figure;
plot(Detrend_Data{1}(:,1));
xlabel("Time [s]");
ylabel("Amplitude [V]");
title("EMG on right bicep femoris");

saveas(fig3,"Detrended EMG R Biceps.png");


fig5 = figure;
plot(Detrend_Data{1}(:,1));
xlabel("Time [s]");
ylabel("Amplitude [V]");
title("EMG on right bicep femoris");

saveas(fig5,"Detrended EMG R Biceps.png");


fig6 = figure;
plot(Filtered_data{1}(:,1));
xlabel("Time [s]");
ylabel("Amplitude [V]");
title("EMG on right bicep femoris");

saveas(fig6,"Filtered EMG R Biceps.png");


fig7 = figure;
plot(Rec_filtered_data{1}(:,1));
xlabel("Time [s]");
ylabel("Amplitude [V]");
title("EMG on right bicep femoris");

saveas(fig7,"Rectified EMG R Biceps.png");
