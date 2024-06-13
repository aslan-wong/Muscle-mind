clear;
close all;
position = [[0,600,400,250];[400,600,400,250];[800,600,400,250];[0,300,400,250];[400,300,400,250];[800,300,400,250];[0,0,400,250];[400,0,400,250];[800,0,400,250]];
maindirs = [...
    "D:\DataSet\EMG_Data\data_MM\action1\user2 action1";"D:\DataSet\EMG_Data\data_MM\action2\user2 action2";"D:\DataSet\EMG_Data\data_MM\action3\user2 action3";"D:\DataSet\EMG_Data\data_MM\action4\user2 action4"; "D:\DataSet\EMG_Data\data_MM\action5\user2 action5"];
path = [];
for i = 1 : length(maindirs)
    path = [path; getpath(maindirs(i))];
end
minid_path = [];
minid_cell = cell(length(path)/9, 5);
muscle_path = [];
muscle_cell = cell(length(path)/9*8, 5);
minid_count = 1;
muscle_count = 1;
fouce = 0;
for i = 1 : length(path)
    p = split(path(i)," ");
    p = split(p(3),"_");
    pp = char(p(1));
    action = str2num(pp(7));
    signal_c = p(3);
    if startsWith(string(signal_c),"mind")
        signal = 0;
    elseif startsWith(string(signal_c),"muscle")
        signal = 1;
    end
    data = readmatrix(path(i));
    data = data(:,2);
    data = data(~isnan(data));
%     plot(data)
%     [b,a] = butter(2,1/200*2,'high');
%     data = filter(b,a, data);
%     plot(data)
%     if startsWith(string(signal_c),"muscle")
%         data=(data-2000)/2000;
%     end
    n = length(data);
    
    if signal == 0
        minid_path = [minid_path;path(i)];
        minid_cell(minid_count,:) = {path(i),data,action,fouce, n};
        minid_count = minid_count + 1;
    else
        muscle_path = [muscle_path;path(i)];
        muscle_cell(muscle_count,:) = {path(i),data,action,fouce, n};
        muscle_count = muscle_count + 1;
        fouce = 1 - fouce;
    end
end


height_enegrys=[];
first = [];
second = [];
Peak = 0;
for i = 2 : length(muscle_cell)/8
    close all;
    n = 10000000;
    for j = 1 : 8
        index = (i-1)*8+j;
        if n > muscle_cell{index,5}
            n = muscle_cell{index,5};
        end
    end
    data = cell2mat(muscle_cell((i-1)*8+1,2));
    muscle_cell((i-1)*8+1,1)
    data = data(1:n);
%     figure('position',position(1,:));plot(data);axis([0,10000,-1,1]);
    for j = 2 : 2
        index = (i-1)*8+j;
        d = cell2mat(muscle_cell(index,2));
        d = d(1:n);
        data = data + d;
%         figure('position',position(j,:));plot(d);axis([0,10000,-1,1]);
    end
%     figure('position',position(9,:));plot(data/8);
    if i == 2
        Peak = max(get_MVC_Peak(data/8, 200));
        continue;
    else
        MVC_Data = get_MVC(data/8, Peak, 200);
        figure;plot(MVC_Data)
    end
%     figure;plot(data/8);hold on;plot(EMG_mask);
end



function path = getpath(maindir)
    subdir  = dir(char(maindir));
    path=[];
    for i = 1 : 1: length( subdir )
        if( isequal( subdir( i ).name, '.' )||...
            isequal( subdir( i ).name, '..'))             % 如果不是目录则跳过
            continue;
        end
        if  ~subdir(i).isdir
            datpath = fullfile( maindir, subdir( i ).name);
            path = [path; string(datpath)];
            continue;
        end
        subdirpath = fullfile( maindir, subdir( i ).name);
        dat = dir( subdirpath );
         for j = 1 : 1:length(dat)
            if( isequal( dat( j ).name, '.' )||...
                isequal( dat( j ).name, '..'))             % 如果不是目录则跳过
                continue;
             end
            datpath = fullfile( maindir, subdir( i ).name, dat( j ).name);
            path = [path; string(datpath)];
         end
    end
end


%     Preprocess the data by segmenting each file into individual EMG using a hysteresis comparator on the signal power
%     
%     Inputs:
%     *x (np.array): EMG signal
%     *fs (float): sampling frequency in Hz 
%     Outputs:
%     *EMGegments (np.array of np.arrays): a list of EMG signal arrays corresponding to each EMG
%     EMG_mask (np.array): an array of booleans that are True at the indices where a EMG is in progress
function [EMGegments, EMG_mask, EMG_start, EMG_end] = segment_EMG_by_heightest(x, fs, varargin)
    position = [[0,600,400,250];[400,600,400,250];[800,600,400,250];[0,300,400,250];[400,300,400,250];[800,300,400,250];[0,0,400,250];[400,0,400,250];[800,0,400,250]];
%     figure('position',position(1,:));plot(x);

    n = length(x);
    EMG_mask = false(n,1);

%     # Segment EMG
    EMGegments = [];
    EMG_start = [];
    EMG_end = [];
    [b,a] = butter(2,1/fs*2,'high');
    env = filter(b,a, x.^2);
    [b,a] = butter(2,10/fs*2,'low');
%     env = filter(b,a,env);
%     figure;plot(env);
    [enegry,height_points] = getenergy(env);
%     figure;plot(enegry);
%     hold on;
%     scatter(b,enegry(b));
%     # Define hysteresis thresholds
    for i = 1 : length(height_points)
        height_index = height_points(i) * 20;
        seg_start = height_index - 250;
        seg_end = height_index + 250;
        if seg_start < 1 
            seg_end = seg_end + (1 - seg_start);
            seg_start = 1;
        end
        if seg_end > n
            seg_start = seg_start - (seg_end  - n);
            seg_end = n;
        end
        if length(EMGegments) == 0
             EMGegments =  x(seg_start: seg_end);
        else
            EMGegments = [EMGegments, x(seg_start: seg_end)];
        end
        EMG_start = [EMG_start, seg_start];
        EMG_end = [EMG_end, seg_end];
        EMG_mask(seg_start: seg_end) = true;
    end

%     hold on;plot(EMG_mask);
end



function [enegry_d, height_points] = getenergy(data)
    n = length(data);
    frame_len = 50;
    step = 20;
    enegry = [];
    height_points = [];
    for i = 1  : step: n - frame_len-1
        frame_enegry = 0;
        for index = 0 : frame_len-1
            frame_enegry = frame_enegry + data(i + index)* data(i + index);
        end
        enegry= [enegry,frame_enegry];
    end
    enegry_d = enegry;
    frame_len = 25;
    step = 20;
    i = 1;
    while i < length(enegry)
        start = i;
        endd = i + frame_len-1;
        if endd > length(enegry)
            endd = length(enegry);
        end
        [~, b] = max(enegry(start : endd));
        height_point = b+start-1;
        
        if length(height_points) < 12 || enegry_d(height_point) > 0.1 
            height_points= [height_points,height_point];
            start = height_point-10;
            endd = height_point + 10;
            if start < 1
                start = 1;
            end
            if endd > n
                endd = n;
            end
            enegry(start : endd) = 0;
            i = height_point + 22;
        else
            i = height_point + 10;
        end
        
    end
    
%         figure;plot(enegry_d);
%         hold on;
%         scatter(height_points,enegry_d(height_points));
end

function Norm_data = get_MVC(data, Peak, fs)
    %Power spectrum
    [P,f] = pspectrum(data,200);
%     plot(f,pow2db(P));
    
    % Detrend data
    Detrend_Data = detrend(data);
    
    
    % Rectify filtered data
    Rec_filtered_data = abs(Detrend_Data);
    
    % Apply moveing average
    Movav_data = movmean(Rec_filtered_data,150,1);
    
    
    % Normalize data
    Norm_data = Movav_data/Peak*100; 
end



function Peak = get_MVC_Peak(data, fs)
    %Power spectrum
    [P,f] = pspectrum(data,fs);
%     plot(f,pow2db(P));
    
    % Detrend data
    Detrend_Data = detrend(data);
    
    
    % Rectify filtered data
    Rec_filtered_data = abs(Detrend_Data);
    
    % Apply moveing average
    Movav_data = movmean(Rec_filtered_data,150,1);
    
    % Find peak MVC
    Peak = max(Movav_data);
end




