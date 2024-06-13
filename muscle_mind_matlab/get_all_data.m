action_test_label_all = action_test_label;
action_train_label_all=action_train_label;
mindeffort_test_label_all=mindeffort_test_label;
mindeffort_train_label_all=mindeffort_train_label;
rm_test_label_all=rm_test_label;
rm_train_label_all=rm_train_label;
test_data_1d_all=test_data_1d;
train_data_1d_all=train_data_1d;

users = ["user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10", "user11", "user13"];
for i = 1 : length(users)
    dir=['D:\项目\muscle_mind项目\muscle_mind_matlab\ML_Feature\', char(users(i))];
    load([dir, '\action_test_label.mat']);
    action_test_label_all = [action_test_label_all, action_test_label];

    load([dir, '\action_train_label.mat']);
    action_train_label_all = [action_train_label_all, action_train_label];

    load([dir, '\mindeffort_test_label.mat']);
    mindeffort_test_label_all = [mindeffort_test_label_all, mindeffort_test_label];

    load([dir, '\mindeffort_train_label.mat']);
    mindeffort_train_label_all = [mindeffort_train_label_all, mindeffort_train_label];

    load([dir, '\rm_test_label.mat']);
    rm_test_label_all = [rm_test_label_all, rm_test_label];

    load([dir, '\rm_train_label.mat']);
    rm_train_label_all = [rm_train_label_all, rm_train_label];

    load([dir, '\test_data_1d.mat']);
    test_data_1d_all = [test_data_1d_all; test_data_1d];

    load([dir, '\train_data_1d.mat']);
    train_data_1d_all = [train_data_1d_all; train_data_1d];


end

%% 

save('action_test_label_all.mat','action_test_label_all');
save('action_train_label_all.mat','action_train_label_all');
save('mindeffort_test_label_all.mat','mindeffort_test_label_all');
save('mindeffort_train_label_all.mat','mindeffort_train_label_all');
save('rm_test_label_all.mat','rm_test_label_all');
save('rm_train_label_all.mat','rm_train_label_all');
save('test_data_1d_all.mat','test_data_1d_all');
save('train_data_1d_all.mat','train_data_1d_all');
