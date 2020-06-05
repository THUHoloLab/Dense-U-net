clc
depth_file_name = 'D:/pycharm/up_down_code/particles_information_extraction/predict_data/depth_0.txt';
x_file_name     = 'D:/pycharm/up_down_code/particles_information_extraction/predict_data/x_0.txt';
y_file_name     = 'D:/pycharm/up_down_code/particles_information_extraction/predict_data/y_0.txt';
r_file_name     = 'D:/pycharm/up_down_code/particles_information_extraction/predict_data/r_0.txt';

depth_label_file_name = 'D:/pycharm/up_down_code/particles_information_extraction/label_data/depth_0.txt';
x_label_file_name     = 'D:/pycharm/up_down_code/particles_information_extraction/label_data/x_0.txt';
y_label_file_name     = 'D:/pycharm/up_down_code/particles_information_extraction/label_data/y_0.txt';
r_label_file_name     = 'D:/pycharm/up_down_code/particles_information_extraction/label_data/r_0.txt';

depth_data = textread(depth_file_name);
%depth_data = (depth_data*2.048/255)*1000+1000;
x_data = textread(x_file_name);
y_data = textread(y_file_name);
r_data = textread(r_file_name);

figure(1);
scatter3(x_data,y_data,depth_data,r_data,'o','red')
hold on
%{
depth_data_label = textread(depth_label_file_name);
depth_data_label = (depth_data_label*2.048/255)*1000+1000;
x_data_label = textread(x_label_file_name);
y_data_label = textread(y_label_file_name);
r_data_label = textread(r_label_file_name);
scatter3(x_data_label,y_data_label,depth_data_label,r_data_label,'black')
%}
figure(2)

depth_data_label = textread(depth_label_file_name);
%depth_data_label = (depth_data_label*2.048/255)*1000+1000;
x_data_label = textread(x_label_file_name);
y_data_label = textread(y_label_file_name);
r_data_label = textread(r_label_file_name);
scatter3(x_data_label,y_data_label,depth_data_label,r_data_label*2,'+','b')

figure(3);
scatter3(x_data,y_data,depth_data,r_data,'o','r')
hold on
depth_data_label = textread(depth_label_file_name);
%depth_data_label = (depth_data_label*2.048/255)*1000+1000;
x_data_label = textread(x_label_file_name);
y_data_label = textread(y_label_file_name);
r_data_label = textread(r_label_file_name);
scatter3(x_data_label,y_data_label,depth_data_label,r_data_label*2,'+','b')
legend({'Ground True','Predict'});