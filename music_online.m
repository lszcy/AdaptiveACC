close all
clear
clc

tic

load("zoneA_6l4m.mat")
load("zoneB_6l4m.mat")
% load('HemiAnechoicRoom_zoneA_6l4m.mat')
% load('HemiAnechoicRoom_zoneB_6l4m.mat')
% load('SmallMeetingRoom_zoneA_m.mat')
% load('SmallMeetingRoom_zoneB_m.mat')


% 本仿真基于8麦克风(各4麦)、6扬声器
% 将所有数据先时域表示
% 滤波器长度、脉冲响应长度
N = 128;
I = 3640;
B = 2*N;

% 数据块长度
% B = I + N;
rng(1000);
x_old = zeros(N,1);
x_w_old = zeros(N,6);
x_w_oldc = zeros(N,6);

[y,fs] = audioread("wodezuguo.wav");
song_start = 72*fs;
song_end = song_start + 30*fs; 
x = y(song_start:song_end-1);
% sound(x,fs)

% 频域扬声器滤波系数
W = zeros(B,6);
% Wc = 1e-6 .* ones(B,6);

% 频域房间脉冲响应（真实值）
fs1 = 4e3;
[inx,f]=freq2bins([0,20000],B,fs1);
F = exp(-1i*2*pi*(0:I-1).*f(:)/fs1);
%% 亮区传递函数变换到频域
H_Bm28 = F * zoneA_m(:,:,1);
% H_Bm28 = F * HemiAnechoicRoom_zoneA_m(:,:,1);
% H_Bm28 = F * SmallMeetingRoom_zoneA_m(:,:,1);
conjm28 = conj(H_Bm28(2:N,:));
conjm28 = flip(conjm28);
H_Bm28 =[H_Bm28;conjm28];

H_Bm29 = F * zoneA_m(:,:,2);
% H_Bm29 = F * HemiAnechoicRoom_zoneA_m(:,:,2);
% H_Bm29 = F * SmallMeetingRoom_zoneA_m(:,:,2);
conjm29 = conj(H_Bm29(2:N,:));
conjm29 = flip(conjm29);
H_Bm29 =[H_Bm29;conjm29];

H_Bm36 = F * zoneA_m(:,:,3);
% H_Bm36 = F * HemiAnechoicRoom_zoneA_m(:,:,3);
% H_Bm36 = F * SmallMeetingRoom_zoneA_m(:,:,3);
conjm36 = conj(H_Bm36(2:N,:));
conjm36 = flip(conjm36);
H_Bm36 =[H_Bm36;conjm36];

H_Bm37 = F * zoneA_m(:,:,4);
% H_Bm37 = F * HemiAnechoicRoom_zoneA_m(:,:,4);
% H_Bm37 = F * SmallMeetingRoom_zoneA_m(:,:,4);
conjm37 = conj(H_Bm37(2:N,:));
conjm37 = flip(conjm37);
H_Bm37 =[H_Bm37;conjm37];
%% 暗区传递函数变换到频域
H_Dm28 = F * zoneB_m(:,:,1);
% H_Dm28 = F * HemiAnechoicRoom_zoneB_m(:,:,1);
% H_Dm28 = F * SmallMeetingRoom_zoneB_m(:,:,1);
conjm28 = conj(H_Dm28(2:N,:));
conjm28 = flip(conjm28);
H_Dm28 =[H_Dm28;conjm28];

H_Dm29 = F * zoneB_m(:,:,2);
% H_Dm29 = F * HemiAnechoicRoom_zoneB_m(:,:,2);
% H_Dm29 = F * SmallMeetingRoom_zoneB_m(:,:,2);
conjm29 = conj(H_Dm29(2:N,:));
conjm29 = flip(conjm29);
H_Dm29 =[H_Dm29;conjm29];

H_Dm36 = F * zoneB_m(:,:,3);
% H_Dm36 = F * HemiAnechoicRoom_zoneB_m(:,:,3);
% H_Dm36 = F * SmallMeetingRoom_zoneB_m(:,:,3);
conjm36 = conj(H_Dm36(2:N,:));
conjm36 = flip(conjm36);
H_Dm36 =[H_Dm36;conjm36];

H_Dm37 = F * zoneB_m(:,:,4);
% H_Dm37 = F * HemiAnechoicRoom_zoneB_m(:,:,4);
% H_Dm37 = F * SmallMeetingRoom_zoneB_m(:,:,4);
conjm37 = conj(H_Dm37(2:N,:));
conjm37 = flip(conjm37);
H_Dm37 =[H_Dm37;conjm37];

%% 调换位置
H_Bm28c = H_Dm28;
H_Bm29c = H_Dm29;
H_Bm36c = H_Dm36;
H_Bm37c = H_Dm37;

H_Dm28c = H_Bm28;
H_Dm29c = H_Bm29;
H_Dm36c = H_Bm36;
H_Dm37c = H_Bm37;

% 频域房间脉冲响应（估计值）
estH_Bm28 = 0.000001 .* ones(B,6);
estH_Bm29 = 0.000001 .* ones(B,6);
estH_Bm36 = 0.000001 .* ones(B,6);
estH_Bm37 = 0.000001 .* ones(B,6);
estH_Dm28 = 0.000001 .* ones(B,6);
estH_Dm29 = 0.000001 .* ones(B,6);
estH_Dm36 = 0.000001 .* ones(B,6);
estH_Dm37 = 0.000001 .* ones(B,6);

estH_Bm28c = 0.000001 .* ones(B,6);
estH_Bm29c = 0.000001 .* ones(B,6);
estH_Bm36c = 0.000001 .* ones(B,6);
estH_Bm37c = 0.000001 .* ones(B,6);
estH_Dm28c = 0.000001 .* ones(B,6);
estH_Dm29c = 0.000001 .* ones(B,6);
estH_Dm36c = 0.000001 .* ones(B,6);
estH_Dm37c = 0.000001 .* ones(B,6);

mean_estH_Bm28 = zeros(B,6);
mean_estH_Bm29 = zeros(B,6);
mean_estH_Bm36 = zeros(B,6);
mean_estH_Bm37 = zeros(B,6);
mean_estH_Dm28 = zeros(B,6);
mean_estH_Dm29 = zeros(B,6);
mean_estH_Dm36 = zeros(B,6);
mean_estH_Dm37 = zeros(B,6);

mean_estH_Bm28c = zeros(B,6);
mean_estH_Bm29c = zeros(B,6);
mean_estH_Bm36c = zeros(B,6);
mean_estH_Bm37c = zeros(B,6);
mean_estH_Dm28c = zeros(B,6);
mean_estH_Dm29c = zeros(B,6);
mean_estH_Dm36c = zeros(B,6);
mean_estH_Dm37c = zeros(B,6);

% 得到处理的数据块数量，向下取整
num_block = floor(length(x) / N);

% 创建 N 阶单位矩阵
I_N = eye(N);
% 创建 N 阶零矩阵
zero_N = zeros(N);
g = [I_N, zero_N; zero_N, zero_N];
k = [zero_N I_N];

% 步长以及相关参数
mu = 0.004; 
alpha = 0.5;
lambda = 1 - alpha;

% 信号功率
p = ones(B,1);
pc = ones(B,1);

% 生成一个2x2的单位矩阵
unitMatrix2x2 = eye(6);
% 使用repmat函数将2x2的单位矩阵复制成2x2x257的三维数组
unitMatrix3D = repmat(unitMatrix2x2, [1, 1, B]);
K = unitMatrix3D;
C = unitMatrix3D;
W1heng = ones(6,B);

% 生成一个2x2的单位矩阵
unitMatrix2x2c = eye(6);
% 使用repmat函数将2x2的单位矩阵复制成2x2x257的三维数组
unitMatrix3Dc = repmat(unitMatrix2x2c, [1, 1, B]);
Kc = unitMatrix3Dc;
Cc = unitMatrix3Dc;
W1hengc = ones(6,B);
% W1heng = unitMatrix3D;

music_online_outputACset1 = [];

all_estH_Bm28 = repmat(estH_Bm28, [1, 1, num_block]);
all_estH_Bm29 = repmat(estH_Bm29, [1, 1, num_block]);
all_estH_Bm36 = repmat(estH_Bm36, [1, 1, num_block]);
all_estH_Bm37 = repmat(estH_Bm37, [1, 1, num_block]);
all_estH_Dm28 = repmat(estH_Dm28, [1, 1, num_block]);
all_estH_Dm29 = repmat(estH_Dm29, [1, 1, num_block]);
all_estH_Dm36 = repmat(estH_Dm36, [1, 1, num_block]);
all_estH_Dm37 = repmat(estH_Dm37, [1, 1, num_block]);

all_estH_Bm28c = repmat(estH_Bm28c, [1, 1, num_block]);
all_estH_Bm29c = repmat(estH_Bm29c, [1, 1, num_block]);
all_estH_Bm36c = repmat(estH_Bm36c, [1, 1, num_block]);
all_estH_Bm37c = repmat(estH_Bm37c, [1, 1, num_block]);
all_estH_Dm28c = repmat(estH_Dm28c, [1, 1, num_block]);
all_estH_Dm29c = repmat(estH_Dm29c, [1, 1, num_block]);
all_estH_Dm36c = repmat(estH_Dm36c, [1, 1, num_block]);
all_estH_Dm37c = repmat(estH_Dm37c, [1, 1, num_block]);

all_Y_Bm28 = [];
all_Y_Dm28c = [];

smooth_num = 20;
forget_factor = 0.99;
for n = 1:num_block
    n
    if n < 5625

    % 将输入信号转换到频域，先由扬声器滤波
    x_n = x((n-1)*N + 1 : n*N);
    x_now = [x_old;x_n];
    X = fft(x_now,B);
    x_old = x_n;

    % 扬声器的输出信号，作为建模滤波器的输入
    X_W = X.*W;
    x_w = real(ifft(X_W));
    x_w = k * x_w;
    x_w_now = [x_w_old;x_w];
    XW = fft(x_w_now,B);
    x_w_old = x_w;

    %% 对比度一帧一帧计算？
    Y_Bm28 = XW.*H_Bm28;
    ybm28 = k * real(ifft(Y_Bm28));
    Y_Bm28 = fft(ybm28,B);
    Y_Bm28 = sum(Y_Bm28,2);
    all_Y_Bm28 = [all_Y_Bm28 Y_Bm28];

    Y_Bm29 = XW.*H_Bm29;
    ybm29 = k * real(ifft(Y_Bm29));
    Y_Bm29 = fft(ybm29,B);
    Y_Bm29 = sum(Y_Bm29,2);

    Y_Bm36 = XW.*H_Bm36;
    ybm36 = k * real(ifft(Y_Bm36));
    Y_Bm36 = fft(ybm36,B);
    Y_Bm36 = sum(Y_Bm36,2);

    Y_Bm37 = XW.*H_Bm37;
    ybm37 = k * real(ifft(Y_Bm37));
    Y_Bm37 = fft(ybm37,B);
    Y_Bm37 = sum(Y_Bm37,2);

    Y_Dm28 = XW.*H_Dm28;
    ydm28 = k * real(ifft(Y_Dm28));
    Y_Dm28 = fft(ydm28,B);
    Y_Dm28 = sum(Y_Dm28,2);

    Y_Dm29 = XW.*H_Dm29;
    ydm29 = k * real(ifft(Y_Dm29));
    Y_Dm29 = fft(ydm29,B);
    Y_Dm29 = sum(Y_Dm29,2);

    Y_Dm36 = XW.*H_Dm36;
    ydm36 = k * real(ifft(Y_Dm36));
    Y_Dm36 = fft(ydm36,B);
    Y_Dm36 = sum(Y_Dm36,2);

    Y_Dm37 = XW.*H_Dm37;
    ydm37 = k * real(ifft(Y_Dm37));
    Y_Dm37 = fft(ydm37,B);
    Y_Dm37 = sum(Y_Dm37,2);

    y_bm28 = k * real(ifft(XW.*estH_Bm28));
    y_bm28 = sum(y_bm28,2);
    y_bm29 = k * real(ifft(XW.*estH_Bm29));
    y_bm29 = sum(y_bm29,2);
    y_bm36 = k * real(ifft(XW.*estH_Bm36));
    y_bm36 = sum(y_bm36,2);
    y_bm37 = k * real(ifft(XW.*estH_Bm37));
    y_bm37 = sum(y_bm37,2);

    y_dm28 = k * real(ifft(XW.*estH_Dm28));
    y_dm28 = sum(y_dm28,2);
    y_dm29 = k * real(ifft(XW.*estH_Dm29));
    y_dm29 = sum(y_dm29,2);
    y_dm36 = k * real(ifft(XW.*estH_Dm36));
    y_dm36 = sum(y_dm36,2);
    y_dm37 = k * real(ifft(XW.*estH_Dm37));
    y_dm37 = sum(y_dm37,2);

    d_bm28 = k * real(ifft(XW.*H_Bm28));
    d_bm28 = sum(d_bm28,2);
    d_bm29 = k * real(ifft(XW.*H_Bm29));
    d_bm29 = sum(d_bm29,2);
    d_bm36 = k * real(ifft(XW.*H_Bm36));
    d_bm36 = sum(d_bm36,2);
    d_bm37 = k * real(ifft(XW.*H_Bm37));
    d_bm37 = sum(d_bm37,2);

    d_dm28 = k * real(ifft(XW.*H_Dm28));
    d_dm28 = sum(d_dm28,2);
    d_dm29 = k * real(ifft(XW.*H_Dm29));
    d_dm29 = sum(d_dm29,2);
    d_dm36 = k * real(ifft(XW.*H_Dm36));
    d_dm36 = sum(d_dm36,2);
    d_dm37 = k * real(ifft(XW.*H_Dm37));
    d_dm37 = sum(d_dm37,2);

    e_bm28 = d_bm28 - y_bm28;
    E_Bm28 = fft(k' * e_bm28,B);
    e_bm29 = d_bm29 - y_bm29;
    E_Bm29 = fft(k' * e_bm29,B);
    e_bm36 = d_bm36 - y_bm36;
    E_Bm36 = fft(k' * e_bm36,B);
    e_bm37 = d_bm37 - y_bm37;
    E_Bm37 = fft(k' * e_bm37,B);

    e_dm28 = d_dm28 - y_dm28;
    E_Dm28 = fft(k' * e_dm28,B);
    e_dm29 = d_dm29 - y_dm29;
    E_Dm29 = fft(k' * e_dm29,B);
    e_dm36 = d_dm36 - y_dm36;
    E_Dm36 = fft(k' * e_dm36,B);
    e_dm37 = d_dm37 - y_dm37;
    E_Dm37 = fft(k' * e_dm37,B);

    p = lambda .* p + alpha .* abs(XW).^2;
    mu_k = mu.* 1./(p);
%         mu_k = mu;
    muXE_Bm28 = mu_k .* conj(XW) .* E_Bm28;
    muXE_Bm29 = mu_k .* conj(XW) .* E_Bm29;
    muXE_Bm36 = mu_k .* conj(XW) .* E_Bm36;
    muXE_Bm37 = mu_k .* conj(XW) .* E_Bm37;
    muXE_Dm28 = mu_k .* conj(XW) .* E_Dm28;
    muXE_Dm29 = mu_k .* conj(XW) .* E_Dm29;
    muXE_Dm36 = mu_k .* conj(XW) .* E_Dm36;
    muXE_Dm37 = mu_k .* conj(XW) .* E_Dm37;

    estH_Bm28 = estH_Bm28 + 2 * fft(g * real(ifft(muXE_Bm28)));
    estH_Bm29 = estH_Bm29 + 2 * fft(g * real(ifft(muXE_Bm29)));
    estH_Bm36 = estH_Bm36 + 2 * fft(g * real(ifft(muXE_Bm36)));
    estH_Bm37 = estH_Bm37 + 2 * fft(g * real(ifft(muXE_Bm37)));
    estH_Dm28 = estH_Dm28 + 2 * fft(g * real(ifft(muXE_Dm28)));
    estH_Dm29 = estH_Dm29 + 2 * fft(g * real(ifft(muXE_Dm29)));
    estH_Dm36 = estH_Dm36 + 2 * fft(g * real(ifft(muXE_Dm36)));
    estH_Dm37 = estH_Dm37 + 2 * fft(g * real(ifft(muXE_Dm37)));

    all_estH_Bm28(:,:,n) = estH_Bm28;
    all_estH_Bm29(:,:,n) = estH_Bm29;
    all_estH_Bm36(:,:,n) = estH_Bm36;
    all_estH_Bm37(:,:,n) = estH_Bm37;
    all_estH_Dm28(:,:,n) = estH_Dm28;
    all_estH_Dm29(:,:,n) = estH_Dm29;
    all_estH_Dm36(:,:,n) = estH_Dm36;
    all_estH_Dm37(:,:,n) = estH_Dm37;

    if n >= smooth_num
        for q = n-(smooth_num-1):n
            mean_estH_Bm28 = mean_estH_Bm28 + forget_factor^(n-q).*all_estH_Bm28(:,:,q);
            mean_estH_Bm28 = mean_estH_Bm28./smooth_num;
            mean_estH_Bm29 = mean_estH_Bm29 + forget_factor^(n-q).*all_estH_Bm29(:,:,q);
            mean_estH_Bm29 = mean_estH_Bm29./smooth_num;
            mean_estH_Bm36 = mean_estH_Bm36 + forget_factor^(n-q).*all_estH_Bm36(:,:,q);
            mean_estH_Bm36 = mean_estH_Bm36./smooth_num;
            mean_estH_Bm37 = mean_estH_Bm37 + forget_factor^(n-q).*all_estH_Bm37(:,:,q);
            mean_estH_Bm37 = mean_estH_Bm37./smooth_num;

            mean_estH_Dm28 = mean_estH_Dm28 + forget_factor^(n-q).*all_estH_Dm28(:,:,q);
            mean_estH_Dm28 = mean_estH_Dm28./smooth_num;
            mean_estH_Dm29 = mean_estH_Dm29 + forget_factor^(n-q).*all_estH_Dm29(:,:,q);
            mean_estH_Dm29 = mean_estH_Dm29./smooth_num;
            mean_estH_Dm36 = mean_estH_Dm36 + forget_factor^(n-q).*all_estH_Dm36(:,:,q);
            mean_estH_Dm36 = mean_estH_Dm36./smooth_num;
            mean_estH_Dm37 = mean_estH_Dm37 + forget_factor^(n-q).*all_estH_Dm37(:,:,q);
            mean_estH_Dm37 = mean_estH_Dm37./smooth_num;
        end
    else
        mean_estH_Bm28 = estH_Bm28;
        mean_estH_Bm29 = estH_Bm29;
        mean_estH_Bm36 = estH_Bm36;
        mean_estH_Bm37 = estH_Bm37;
        mean_estH_Dm28 = estH_Dm28;
        mean_estH_Dm29 = estH_Dm29;
        mean_estH_Dm36 = estH_Dm36;
        mean_estH_Dm37 = estH_Dm37;
    end

    XB = cat(3,mean_estH_Bm28,mean_estH_Bm29,mean_estH_Bm36,mean_estH_Bm37);
    XD = cat(3,mean_estH_Dm28,mean_estH_Dm29,mean_estH_Dm36,mean_estH_Dm37);
    %     XB = cat(3,estH_Bm28,estH_Bm29,estH_Bm36,estH_Bm37);
    %     XD = cat(3,estH_Dm28,estH_Dm29,estH_Dm36,estH_Dm37);
    outputAC = 10*log10((norm(Y_Bm28)^2 + norm(Y_Bm29)^2 + norm(Y_Bm36)^2 + norm(Y_Bm37)^2)/(norm(Y_Dm28)^2 + norm(Y_Dm29)^2 + norm(Y_Dm36)^2 + norm(Y_Dm37)^2))
    music_online_outputACset1 = [music_online_outputACset1 outputAC];

    % 可以只计算前半部分，后半部分共轭对称
    kbins = B;


    for  kb = 1:kbins
        % 调用函数
        xk = XB(kb,:,:);
        xk = squeeze(xk);
        nk = XD(kb,:,:);
        nk = squeeze(nk);
        Kk = K(:,:,kb);
        Ck = C(:,:,kb);
        W1hengk = W1heng(:,kb);
        [Kkk,Ckk,W1hengkk,W_kbins] = fget6l4m(Kk,Ck,W1hengk,xk,nk);
        K(:,:,kb) = Kkk;
        C(:,:,kb) = Ckk;
        W1heng(:,kb) = W1hengkk;
        W(kb,:) = W_kbins';

    end
    arrayEffort = 10*log10(norm(W).^2)
    Wc = W;
else
    % 将输入信号转换到频域，先由扬声器滤波
    x_n = x((n-1)*N + 1 : n*N);
    x_now = [x_old;x_n];
    X = fft(x_now,B);
    x_old = x_n;

    % 扬声器的输出信号，作为建模滤波器的输入
    X_Wc = X.*Wc;
    x_wc = real(ifft(X_Wc));
    x_wc = k * x_wc;
    x_w_nowc = [x_w_oldc;x_wc];
    XWc = fft(x_w_nowc,B);
    x_w_oldc = x_wc;

    %% 对比度一帧一帧计算？
    Y_Bm28c = XWc.*H_Bm28c;
    ybm28c = k * real(ifft(Y_Bm28c));
    Y_Bm28c = fft(ybm28c,B);
    Y_Bm28c = sum(Y_Bm28c,2);

    Y_Bm29c = XWc.*H_Bm29c;
    ybm29c = k * real(ifft(Y_Bm29c));
    Y_Bm29c = fft(ybm29c,B);
    Y_Bm29c = sum(Y_Bm29c,2);

    Y_Bm36c = XWc.*H_Bm36c;
    ybm36c = k * real(ifft(Y_Bm36c));
    Y_Bm36c = fft(ybm36c,B);
    Y_Bm36c = sum(Y_Bm36c,2);

    Y_Bm37c = XWc.*H_Bm37c;
    ybm37c = k * real(ifft(Y_Bm37c));
    Y_Bm37c = fft(ybm37c,B);
    Y_Bm37c = sum(Y_Bm37c,2);

    Y_Dm28c = XWc.*H_Dm28c;
    ydm28c = k * real(ifft(Y_Dm28c));
    Y_Dm28c = fft(ydm28c,B);
    Y_Dm28c = sum(Y_Dm28c,2);
    all_Y_Dm28c = [all_Y_Dm28c Y_Dm28c];

    Y_Dm29c = XWc.*H_Dm29c;
    ydm29c = k * real(ifft(Y_Dm29c));
    Y_Dm29c = fft(ydm29c,B);
    Y_Dm29c = sum(Y_Dm29c,2);

    Y_Dm36c = XWc.*H_Dm36c;
    ydm36c = k * real(ifft(Y_Dm36c));
    Y_Dm36c = fft(ydm36c,B);
    Y_Dm36c = sum(Y_Dm36c,2);

    Y_Dm37c = XWc.*H_Dm37c;
    ydm37c = k * real(ifft(Y_Dm37c));
    Y_Dm37c = fft(ydm37c,B);
    Y_Dm37c = sum(Y_Dm37c,2);

    y_bm28c = k * real(ifft(XWc.*estH_Bm28c));
    y_bm28c = sum(y_bm28c,2);
    y_bm29c = k * real(ifft(XWc.*estH_Bm29c));
    y_bm29c = sum(y_bm29c,2);
    y_bm36c = k * real(ifft(XWc.*estH_Bm36c));
    y_bm36c = sum(y_bm36c,2);
    y_bm37c = k * real(ifft(XWc.*estH_Bm37c));
    y_bm37c = sum(y_bm37c,2);

    y_dm28c = k * real(ifft(XWc.*estH_Dm28c));
    y_dm28c = sum(y_dm28c,2);
    y_dm29c = k * real(ifft(XWc.*estH_Dm29c));
    y_dm29c = sum(y_dm29c,2);
    y_dm36c = k * real(ifft(XWc.*estH_Dm36c));
    y_dm36c = sum(y_dm36c,2);
    y_dm37c = k * real(ifft(XWc.*estH_Dm37c));
    y_dm37c = sum(y_dm37c,2);

    d_bm28c = k * real(ifft(XWc.*H_Bm28c));
    d_bm28c = sum(d_bm28c,2);
    d_bm29c = k * real(ifft(XWc.*H_Bm29c));
    d_bm29c = sum(d_bm29c,2);
    d_bm36c = k * real(ifft(XWc.*H_Bm36c));
    d_bm36c = sum(d_bm36c,2);
    d_bm37c = k * real(ifft(XWc.*H_Bm37c));
    d_bm37c = sum(d_bm37c,2);

    d_dm28c = k * real(ifft(XWc.*H_Dm28c));
    d_dm28c = sum(d_dm28c,2);
    d_dm29c = k * real(ifft(XWc.*H_Dm29c));
    d_dm29c = sum(d_dm29c,2);
    d_dm36c = k * real(ifft(XWc.*H_Dm36c));
    d_dm36c = sum(d_dm36c,2);
    d_dm37c = k * real(ifft(XWc.*H_Dm37c));
    d_dm37c = sum(d_dm37c,2);

    e_bm28c = d_bm28c - y_bm28c;
    E_Bm28c = fft(k' * e_bm28c,B);
    e_bm29c = d_bm29c - y_bm29c;
    E_Bm29c = fft(k' * e_bm29c,B);
    e_bm36c = d_bm36c - y_bm36c;
    E_Bm36c = fft(k' * e_bm36c,B);
    e_bm37c = d_bm37c - y_bm37c;
    E_Bm37c = fft(k' * e_bm37c,B);

    e_dm28c = d_dm28c - y_dm28c;
    E_Dm28c = fft(k' * e_dm28c,B);
    e_dm29c = d_dm29c - y_dm29c;
    E_Dm29c = fft(k' * e_dm29c,B);
    e_dm36c = d_dm36c - y_dm36c;
    E_Dm36c = fft(k' * e_dm36c,B);
    e_dm37c = d_dm37c - y_dm37c;
    E_Dm37c = fft(k' * e_dm37c,B);

    pc = lambda .* pc + alpha .* abs(XWc).^2;
    mu_kc = mu.* 1./(pc);
%     mu_kc = mu;
    muXE_Bm28c = mu_kc .* conj(XWc) .* E_Bm28c;
    muXE_Bm29c = mu_kc .* conj(XWc) .* E_Bm29c;
    muXE_Bm36c = mu_kc .* conj(XWc) .* E_Bm36c;
    muXE_Bm37c = mu_kc .* conj(XWc) .* E_Bm37c;
    muXE_Dm28c = mu_kc .* conj(XWc) .* E_Dm28c;
    muXE_Dm29c = mu_kc .* conj(XWc) .* E_Dm29c;
    muXE_Dm36c = mu_kc .* conj(XWc) .* E_Dm36c;
    muXE_Dm37c = mu_kc .* conj(XWc) .* E_Dm37c;

    estH_Bm28c = estH_Bm28c + 2 * fft(g * real(ifft(muXE_Bm28c)));
    estH_Bm29c = estH_Bm29c + 2 * fft(g * real(ifft(muXE_Bm29c)));
    estH_Bm36c = estH_Bm36c + 2 * fft(g * real(ifft(muXE_Bm36c)));
    estH_Bm37c = estH_Bm37c + 2 * fft(g * real(ifft(muXE_Bm37c)));
    estH_Dm28c = estH_Dm28c + 2 * fft(g * real(ifft(muXE_Dm28c)));
    estH_Dm29c = estH_Dm29c + 2 * fft(g * real(ifft(muXE_Dm29c)));
    estH_Dm36c = estH_Dm36c + 2 * fft(g * real(ifft(muXE_Dm36c)));
    estH_Dm37c = estH_Dm37c + 2 * fft(g * real(ifft(muXE_Dm37c)));

    all_estH_Bm28c(:,:,n) = estH_Bm28c;
    all_estH_Bm29c(:,:,n) = estH_Bm29c;
    all_estH_Bm36c(:,:,n) = estH_Bm36c;
    all_estH_Bm37c(:,:,n) = estH_Bm37c;
    all_estH_Dm28c(:,:,n) = estH_Dm28c;
    all_estH_Dm29c(:,:,n) = estH_Dm29c;
    all_estH_Dm36c(:,:,n) = estH_Dm36c;
    all_estH_Dm37c(:,:,n) = estH_Dm37c;

    if n >= smooth_num
        for q = n-(smooth_num-1):n
            mean_estH_Bm28c = mean_estH_Bm28c + forget_factor^(n-q).*all_estH_Bm28c(:,:,q);
            mean_estH_Bm28c = mean_estH_Bm28c./smooth_num;
            mean_estH_Bm29c = mean_estH_Bm29c + forget_factor^(n-q).*all_estH_Bm29c(:,:,q);
            mean_estH_Bm29c = mean_estH_Bm29c./smooth_num;
            mean_estH_Bm36c = mean_estH_Bm36c + forget_factor^(n-q).*all_estH_Bm36c(:,:,q);
            mean_estH_Bm36c = mean_estH_Bm36c./smooth_num;
            mean_estH_Bm37c = mean_estH_Bm37c + forget_factor^(n-q).*all_estH_Bm37c(:,:,q);
            mean_estH_Bm37c = mean_estH_Bm37c./smooth_num;

            mean_estH_Dm28c = mean_estH_Dm28c + forget_factor^(n-q).*all_estH_Dm28c(:,:,q);
            mean_estH_Dm28c = mean_estH_Dm28c./smooth_num;
            mean_estH_Dm29c = mean_estH_Dm29c + forget_factor^(n-q).*all_estH_Dm29c(:,:,q);
            mean_estH_Dm29c = mean_estH_Dm29c./smooth_num;
            mean_estH_Dm36c = mean_estH_Dm36c + forget_factor^(n-q).*all_estH_Dm36c(:,:,q);
            mean_estH_Dm36c = mean_estH_Dm36c./smooth_num;
            mean_estH_Dm37c = mean_estH_Dm37c + forget_factor^(n-q).*all_estH_Dm37c(:,:,q);
            mean_estH_Dm37c = mean_estH_Dm37c./smooth_num;
        end
    else
        mean_estH_Bm28c = estH_Bm28c;
        mean_estH_Bm29c = estH_Bm29c;
        mean_estH_Bm36c = estH_Bm36c;
        mean_estH_Bm37c = estH_Bm37c;
        mean_estH_Dm28c = estH_Dm28c;
        mean_estH_Dm29c = estH_Dm29c;
        mean_estH_Dm36c = estH_Dm36c;
        mean_estH_Dm37c = estH_Dm37c;
    end

    XBc = cat(3,mean_estH_Bm28c,mean_estH_Bm29c,mean_estH_Bm36c,mean_estH_Bm37c);
    XDc = cat(3,mean_estH_Dm28c,mean_estH_Dm29c,mean_estH_Dm36c,mean_estH_Dm37c);
 
    outputAC = 10*log10((norm(Y_Bm28c)^2 + norm(Y_Bm29c)^2 + norm(Y_Bm36c)^2 + norm(Y_Bm37c)^2)/(norm(Y_Dm28c)^2 + norm(Y_Dm29c)^2 + norm(Y_Dm36c)^2 + norm(Y_Dm37c)^2))
    music_online_outputACset1 = [music_online_outputACset1 outputAC];

    % 可以只计算前半部分，后半部分共轭对称
    kbins = B;

    for  kb = 1:kbins
        % 调用函数
        xkc = XBc(kb,:,:);
        xkc = squeeze(xkc);
        nkc = XDc(kb,:,:);
        nkc = squeeze(nkc);
        Kkc = Kc(:,:,kb);
        Ckc = Cc(:,:,kb);
        W1hengkc = W1hengc(:,kb);
        [Kkkc,Ckkc,W1hengkkc,W_kbinsc] = fget6l4m(Kkc,Ckc,W1hengkc,xkc,nkc);
        Kc(:,:,kb) = Kkkc;
        Cc(:,:,kb) = Ckkc;
        W1hengc(:,kb) = W1hengkkc;
        Wc(kb,:) = W_kbinsc';

    end
    arrayEffort = 10*log10(norm(Wc).^2)
    end
end
toc
% save('online_outputAC.mat','outputACset')
figure(1)
plot(1:n,music_online_outputACset1)

figure(2)
subplot(211)
plot(1:B,real(H_Bm28c(1:B,1)));hold on
plot(1:B,real(estH_Bm28c(1:B,1)));
subplot(212)
plot(1:B,real(H_Bm28c(1:B,1)) - real(estH_Bm28c(1:B,1)));
