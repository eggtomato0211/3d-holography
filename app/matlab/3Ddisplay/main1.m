clear all
close all

i = sqrt(-1);

wav_len = 532.0*10^-9;              % 光源の波長 [m]
Nx = 1024;                     % x方向の画素数 [pixel]
Ny = 1024;                     % y方向の画素数 [pixel]
dx = 3.45*10^-6;                  % LCOSのメッシュサイズ [m]
dy = dx;
wav_num = 2*pi/wav_len;             % 波数 [1/m]

x1 = -Nx/2;
x2 = Nx/2-1;
y1 = -Ny/2;;
y2 = Ny/2-1;
[Fx,Fy]=meshgrid(x1:1:x2,y1:1:y2);

%画像データの読み込み
data01 = double(imread('am.bmp','bmp'));
data01 = imresize(data01,4);
[sizex,sizey]=size(data01)
figure(1);
imshow(data01,[]);

data02 = double(imread('bm.bmp','bmp'));
data02 = imresize(data02,4);
[sizex,sizey]=size(data02)
figure(2);
imshow(data02,[]);

%ランダム位相分布
initial_phase1 = (rand(sizex,sizey)-0.5)*2.0*2.0*pi;
initial_phase2 = (rand(sizex,sizey)-0.5)*2.0*2.0*pi;

%入力画像
input1 = data01.*exp(i*initial_phase1);
input2 = data02.*exp(i*initial_phase2);

d1 = 50.0;
d2 = 100.0;
%nearpropCONVは光波の空間伝搬を計算するフレネル伝搬計算
%物体１(am.bmp)はSLMから距離d1の位置にある。物体２(bm.bmp)はSLMから距離d2の位置にある。
output1 = nearpropCONV(input1,Nx,Ny,dx,dy,0,0,wav_len,d1); 
output2 = nearpropCONV(input2,Nx,Ny,dx,dy,0,0,wav_len,d2); 

%2つの光波の加算
output = output1+output2;

%振幅と位相分布の計算
%phase_outputがSLMに表示する位相分布
amplitude_output = abs(output);
phase_output = angle(output);
figure(4);
imshow(phase_output,[]);

%再生計算
SLM_data = exp(i*phase_output);

%距離-d1での再生像
reconst1 = nearpropCONV(SLM_data,Nx,Ny,dx,dy,0,0,wav_len,-1.0*d1); 
figure(5);
imshow(abs(reconst1),[]);

%距離-d2での再生像
reconst2 = nearpropCONV(SLM_data,Nx,Ny,dx,dy,0,0,wav_len,-1.0*d2); 
figure(6);
imshow(abs(reconst2),[]);



