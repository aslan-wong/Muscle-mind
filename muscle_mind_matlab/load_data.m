clear;
close all;
position = [[0,600,400,250];[400,600,400,250];[800,600,400,250];[0,300,400,250];[400,300,400,250];[800,300,400,250];[0,0,400,250];[400,0,400,250];[800,0,400,250]];
maindirs = [...
    "D:\DataSet\EMG_Data\data_MM\action2\user6 action2";"D:\DataSet\EMG_Data\data_MM\action2\user1 action2";"D:\DataSet\EMG_Data\data_MM\action3\user1 action3";"D:\DataSet\EMG_Data\data_MM\action4\user1 action4"; "D:\DataSet\EMG_Data\data_MM\action5\user1 action5"];
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
    if startsWith(string(signal_c),"muscle")
        data=(data-2048)/2048;
    end
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
train_data = zeros(501,8);
test_data = zeros(501,8);
val_data = zeros(501,8);
action_train_label = []; % 0 1 2 3 4
mindeffort_train_label = []; %0 effort; 1 no effort
rm_train_label = []; % 0 0% ;1 67%; 2 87%

action_test_label = []; % 0 1 2 3 4
mindeffort_test_label = []; %0 effort; 1 no effort
rm_test_label = []; % 0 0% ;1 67%; 2 87%

action_val_label = []; % 0 1 2 3 4
mindeffort_val_label = []; %0 effort; 1 no effort
rm_val_label = []; % 0 0% ;1 67%; 2 87%

action_train_label = []; % 0 1 2 3 4
mindeffort_train_label = []; %0 effort; 1 no effort
rm_train_label = []; % 0 0% ;1 67%; 2 87%


height_enegrys=[];
for i = 1 : length(muscle_cell)/8
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
    for j = 2 : 8
        index = (i-1)*8+j;
        d = cell2mat(muscle_cell(index,2));
        d = d(1:n);
        data = data + d;
%         figure('position',position(j,:));plot(d);axis([0,10000,-1,1]);
    end
    plot(data)
%     figure('position',position(9,:));plot(data/8);
    
    [b,a] = butter(2,1/200*2,'high');
    data = (filter(b,a, data));
    [EMGegments, EMG_mask, EMG_start, EMG_end] = segment_EMG_by_heightest(data/8, 200);
    example = EMGegments(:,5);
    plot(example)
    save('data_example\6.mat',"example")
    for k = 1 : size(EMGegments,2)
        if k > 10
            break;
        end
        one_data=zeros(501, 8);
        for j = 1 : 8
            data = cell2mat(muscle_cell((i-1)*8+j,2));
            one_data(:,j) = data(EMG_start(k): EMG_end(k));
        end
        if k == 3 || k == 9 || k ==6
            test_data = cat(3,test_data, one_data);
            action_test_label = [action_test_label, cell2mat(muscle_cell((i-1)*8+j,3))-1];
            mindeffort_test_label = [mindeffort_test_label, 1-mod(i,2)];
            rm_test_label = [rm_test_label,floor(mod(i-1,6)/2)];
%         elseif k == 6 || k == 12
%             val_data = cat(3,val_data, one_data);
%             action_val_label = [action_val_label, cell2mat(muscle_cell((i-1)*8+j,3))-1];
%             mindeffort_val_label = [mindeffort_val_label,1-mod(i,2)];
%             rm_val_label = [rm_val_label,floor(mod(i-1,6)/2)];
        else
            train_data = cat(3,train_data, one_data);
            action_train_label = [action_train_label, cell2mat(muscle_cell((i-1)*8+j,3))-1];
            mindeffort_train_label = [mindeffort_train_label, 1-mod(i,2)];
            rm_train_label = [rm_train_label,floor(mod(i-1,6)/2)];
        end
        
    end
%     figure;plot(data/8);hold on;plot(EMG_mask);
end

train_data = train_data(:,:,2:end);
test_data = test_data(:,:,2:end);
val_data = val_data(:,:,2:end);
save('train_data.mat','train_data')
save('test_data.mat','test_data')
save('val_data.mat','val_data')
save('action_train_label.mat','action_train_label')
save('mindeffort_train_label.mat','mindeffort_train_label')
save('rm_train_label.mat','rm_train_label')
save('action_val_label.mat','action_val_label')
save('mindeffort_val_label.mat','mindeffort_val_label')
save('rm_val_label.mat','rm_val_label')
save('action_test_label.mat','action_test_label')
save('mindeffort_test_label.mat','mindeffort_test_label')
save('rm_test_label.mat','rm_test_label');



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

%     Preprocess the data by segmenting each file into individual EMG using a hysteresis comparator on the signal power
%     
%     Inputs:
%     *x (np.array): EMG signal
%     *fs (float): sampling frequency in Hz
%     *EMG_padding (float): number of seconds added to the beginning and end of each detected EMG to make sure EMG are not cut short
%     *min_EMG_length (float): length of the minimum possible segment that can be considered a EMG
%     *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
%     *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
%     
%     Outputs:
%     *EMGegments (np.array of np.arrays): a list of EMG signal arrays corresponding to each EMG
%     EMG_mask (np.array): an array of booleans that are True at the indices where a EMG is in progress
function [EMGegments, EMG_mask] = segment_EMG_by_threshold(x, fs, varargin)
    position = [[0,600,400,250];[400,600,400,250];[800,600,400,250];[0,300,400,250];[400,300,400,250];[800,300,400,250];[0,0,400,250];[400,0,400,250];[800,0,400,250]];
%     figure('position',position(1,:));plot(x);
    win = 20;
    orinal_x = x;
%     enegry = getenergy(x);
%     figure;plot(enegry);
%     for i = win : length(x)
%         x(i) = mean(orinal_x(i-win+1:i-1)); 
%     end
%     figure('position',position(2,:));plot(x);
    p = inputParser;            % 函数的输入解析器
    addParameter(p,'EMG_padding',0.2);
    addParameter(p,'min_EMG_len',1); 
    addParameter(p,'max_EMG_len',4); 
    addParameter(p,'th_l_multiplier',0.5); 
    addParameter(p,'th_h_multiplier',0.5); 
    parse(p,varargin{:});% 对输入变量进行解析，如果检测到前面的变量被赋值，则更新变量取值
    EMG_padding = p.Results.EMG_padding;
    max_EMG_len = p.Results.max_EMG_len;
    min_EMG_len = p.Results.min_EMG_len;
    th_l_multiplier = p.Results.th_l_multiplier;
    th_h_multiplier = p.Results.th_h_multiplier;
    n = length(x);
    EMG_mask = false(n,1);

    [b,a] = butter(2,1/fs*2,'high');
    env = filter(b,a, x.^2);
    [b,a] = butter(2,10/fs*2,'low');
    env = filter(b,a,env);
%     env = env - mean(env);
    figure;plot(env);
  
%     # Define hysteresis thresholds
    rms = sqrt(mean(env.^2));
    seg_th_l = th_l_multiplier * rms;
    seg_th_h = th_h_multiplier * rms;

%     # Segment EMG
    EMGegments = [];
    padding = round(fs * EMG_padding);
    min_EMG_samples = round(fs * min_EMG_len);
    max_EMG_samples = round(fs * max_EMG_len);
    EMG_start = 1;
    EMG_end = 1;
    EMG_in_progress = false;
    tolerance = round(0.1 * fs);
    below_th_counter = 0;

    for i = 1 : length(env)
        sample = abs(env(i));
        if EMG_in_progress == true
            if sample < seg_th_l
                below_th_counter = below_th_counter + 1;
                if below_th_counter > tolerance
                    EMG_end = i + padding ;
                    if i + padding > length(x)
                        EMG_end = length(x);
                    end
                    EMG_in_progress = false;
                    if EMG_end + 1 - EMG_start  > min_EMG_samples&& EMG_end + 1 - EMG_start  < max_EMG_samples
                        subdata = x(EMG_start:EMG_end);
                        subdata=[subdata ; zeros(800-length(subdata),1)];
                        if length(EMGegments) == 0
                            EMGegments = subdata;
                        else
                            EMGegments = [EMGegments, subdata];
                        end
                        EMG_mask(EMG_start:EMG_end) = true;
                    end
                end
            elseif i == length(x)
                EMG_end = i;
                EMG_in_progress = false;
                if EMG_end + 1 - EMG_start > min_EMG_samples && EMG_end + 1 - EMG_start < max_EMG_samples
                    subdata = x(EMG_start:EMG_end);
                    subdata=[subdata ; zeros(800-length(subdata),1)];
                    if length(EMGegments) == 0
                        EMGegments = subdata;
                    else
                        EMGegments = [EMGegments, subdata];
                    end
                    EMG_mask(EMG_start:EMG_end) = true;
                end
           else
                below_th_counter = 0;
           end
        else
            if sample > seg_th_h
                EMG_start = i - padding * 3;
                if i  - padding * 3 < 1
                    EMG_start = 1;
                end
                EMG_in_progress = true;
            end
        end
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

