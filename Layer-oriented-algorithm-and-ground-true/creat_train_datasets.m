%{
Nx            -- Number of voxels along voxel voume x axis
Ny            -- Number of voxels along voxel volume y axis
zmax          -- Number of voxels along voxel volume z axis
pixel_size    -- size in microns
particle_size -- diameter of particle in microns
lambda        -- wavelength of light in microns from the laser source
working_dist  -- distance between voxel and camera 1 along z direction
depth_factor  -- voxel length in z-direction in microns
mag           -- magnification
%}
%% Initial value setting
clc
clear all
Nx=512;   %pixels along x
Ny=512;   %pixels along y
mag=5;   
pixel_size=2*mag;%microns
lambda=532*10^-3; %microns
zmax=256;         %pixels
working_dist=1000;   %microns
depth_factor=2^11/zmax; % scaling, now range is set as 2.048cm,2^11microns 8um
%% generation hologram and corresponding ground true
data_cale = 10000;%datasets numbers
data_count = 1;
while data_count < data_cale
 r_out = 0;% particle radius ground true
 x_out = 0;% x ground true
 y_out =0 ;% y ground true
 depth = 0;% z ground true
%Simulating holograms
particle_number=randi(200);%particle number
sidex=Nx*pixel_size;
sidey=Nx*pixel_size;
pattern=zeros(Nx,Ny);
particle_size= randi([5,10],1,particle_number); %particle radius
object_mask=ones(Nx,Ny,particle_number);%Generate multiple particle planes
object_mask_out = ones(Nx,Ny);
pattern_size=2*(particle_size+5);
count=0;
iota=sqrt(-1);
%% Particles are generated in each plane
while count<particle_number
    X = randi([particle_size(count+1),Nx-particle_size(count+1)],1);
    Y = randi([particle_size(count+1),Ny-particle_size(count+1)],1);
    Xmax=min(X+pattern_size(count+1)-1,Nx-particle_size(count+1));
    Ymax=min(Y+pattern_size(count+1)-1,Ny-particle_size(count+1));
    if(pattern(X:Xmax,Y:Ymax)==zeros(Xmax-X+1,Ymax-Y+1))
        pattern(X:Xmax,Y:Ymax)=ones(Xmax-X+1,Ymax-Y+1);
        count=count+1;
        depth(count)=working_dist+randi(zmax)*depth_factor;
        
        numpoints=pattern_size(count);
        
        for theta = 0:2*pi/numpoints:pi-2*pi/numpoints
            xcirc=int16(particle_size(count)/2*cos(theta));
            ycirc=int16(particle_size(count)/2*sin(theta));
            Xmin=max(1,X-xcirc);
            Xmax=min(X+xcirc,Nx-particle_size(count));
            Ymin=max(1,Y-ycirc);
            Ymax=min(Y+ycirc,Ny-particle_size(count));
           
           
            object_mask(Xmin:Xmax,Ymin:Ymax,count) = 0;
            object_mask_out = object_mask_out + object_mask(:,:,count);
            figure(3)
            imagesc(object_mask(:,:,count));
           
        end
        r_out(count) = particle_size(count);
        x_out(count) = X;
        y_out(count) = Y;
%    object_mask(X:Xmax,Y:Ymax,count)=ones(Xmax-X+1,Ymax-Y+1); %square
    end
end
%% Generate hologram
for y=1:1:Ny
    for x=1:1:Nx
        phasemap(x,y)=2*pi/lambda*(1-((y-Ny/2-1)*lambda/sidey)^2-((x-Nx/2-1)*lambda/sidex)^2)^0.5;
%           f1(x,y) = exp(iota*pi*(x + y));
    end
end
phasemap=ifftshift(phasemap);
hologram=zeros(Nx,Ny);
hologram_recontruct=ones(Nx,Ny);
hologram2=zeros(Nx,Ny);
for j=1:1:particle_number
        prop(:,:,j) = exp(-iota*depth(j)*phasemap);
     
 prop_fft = fft2(object_mask(:,:,j));
 U  = ifft2(prop_fft.*prop(:,:,j));
hologram = hologram+abs(U).^2;
end
noisy_hologram=hologram+max(max(hologram))/100*rand(Nx,Ny);
I_new=mat2gray(noisy_hologram);
figure(8)
imshow(I_new);
filename=['./datasets/train_data/data/data/',num2str(data_count),'.tif'];
imwrite(I_new,filename,'tif');
%% Generate ground true
NEW_rgb=zeros(Nx,Ny);
z_max = max(depth);
z_min = min(depth);
z_change = 255/(z_max - z_min);
particle_number = length(r_out);
filename_x=['./datasets/ground_true_text/x_',num2str(data_count),'.txt'];
filename_y=['./datasets/ground_true_text/y_',num2str(data_count),'.txt'];
filename_r=['./datasets/ground_true_text/r_',num2str(data_count),'.txt'];
filename_d=['./datasets/ground_true_text/z_',num2str(data_count),'.txt'];
dlmwrite(filename_x,x_out,'delimiter','\t','newline','pc');
dlmwrite(filename_y,y_out,'delimiter','\t','newline','pc');
dlmwrite(filename_r,r_out,'delimiter','\t','newline','pc');
dlmwrite(filename_d,depth,'delimiter','\t','newline','pc');
for judge = 1:1:particle_number
    for particle_num = 1:1:particle_number

     for x = 1:1:Nx
        for y = 1:1:Ny
            if  y == y_out(particle_num) && x == x_out(particle_num)
                x_new = x - round(r_out(particle_num)/2);
                y_new = y - round(r_out(particle_num)/2);
                
                if y_new < 1
                    y_new = 1;
                end 
                if x_new < 1
                    x_new = 1;
                end
                for x_2_new = x_new:1:x_new+r_out(particle_num)
                    for y_2_new = y_new:1:y_new+r_out(particle_num)
                        
                        NEW_rgb(x_2_new,y_2_new) = (depth(particle_num)-1000)*z_change;
                    end
                end 
               
            end
        end
        
     end
    end
    
end
%save label data
I_new_1=mat2gray(NEW_rgb);
figure(2)
imshow(I_new_1);
filename_1=['./datasets/ground_true/label/label/',num2str(data_count),'.tif'];
imwrite(I_new_1,filename_1,'tif');
data_count = data_count +1;
end


