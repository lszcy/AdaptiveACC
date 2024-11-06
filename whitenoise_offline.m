close all
clear
clc

tic

load("zoneA_6l4m.mat")
load("zoneB_6l4m.mat")

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

% 扬声器的时域输入信号
fs = 48e3;
sigLen = 30*fs;
x = randn(sigLen,1);

% 频域扬声器滤波系数
W = zeros(B,6);
% Wc = 1e-6 .* ones(B,6);
% 频域房间脉冲响应（真实值）
fs1 = 4e3;
[inx,f]=freq2bins([0,8000],B,fs1);
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

estH_Bm28 = randn(B,6);
estH_Bm29 = randn(B,6);
estH_Bm36 = randn(B,6);
estH_Bm37 = randn(B,6);
estH_Dm28 = randn(B,6);
estH_Dm29 = randn(B,6);
estH_Dm36 = randn(B,6);
estH_Dm37 = randn(B,6);
% 移动后传递函数未知，便随机产生
estH_Bm28c = randn(B,6);
estH_Bm29c = randn(B,6);
estH_Bm36c = randn(B,6);
estH_Bm37c = randn(B,6);
estH_Dm28c = randn(B,6);
estH_Dm29c = randn(B,6);
estH_Dm36c = randn(B,6);
estH_Dm37c = randn(B,6);

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

offline_outputAC = [];

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

% 不在线建模，传递函数是预先测得的
    all_estH_Bm28(:,:,n) = H_Bm28;
    all_estH_Bm29(:,:,n) = H_Bm29;
    all_estH_Bm36(:,:,n) = H_Bm36;
    all_estH_Bm37(:,:,n) = H_Bm37;
    all_estH_Dm28(:,:,n) = H_Dm28;
    all_estH_Dm29(:,:,n) = H_Dm29;
    all_estH_Dm36(:,:,n) = H_Dm36;
    all_estH_Dm37(:,:,n) = H_Dm37;

    if n >= smooth_num
        for q = n-smooth_num+1:n
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
        mean_estH_Bm28 = H_Bm28;
        mean_estH_Bm29 = H_Bm29;
        mean_estH_Bm36 = H_Bm36;
        mean_estH_Bm37 = H_Bm37;
        mean_estH_Dm28 = H_Dm28;
        mean_estH_Dm29 = H_Dm29;
        mean_estH_Dm36 = H_Dm36;
        mean_estH_Dm37 = H_Dm37;
    end

    XB = cat(3,mean_estH_Bm28,mean_estH_Bm29,mean_estH_Bm36,mean_estH_Bm37);
    XD = cat(3,mean_estH_Dm28,mean_estH_Dm29,mean_estH_Dm36,mean_estH_Dm37);
    %     XB = cat(3,estH_Bm28,estH_Bm29,estH_Bm36,estH_Bm37);
    %     XD = cat(3,estH_Dm28,estH_Dm29,estH_Dm36,estH_Dm37);
    outputAC = 10*log10((norm(Y_Bm28)^2 + norm(Y_Bm29)^2 + norm(Y_Bm36)^2 + norm(Y_Bm37)^2)/(norm(Y_Dm28)^2 + norm(Y_Dm29)^2 + norm(Y_Dm36)^2 + norm(Y_Dm37)^2))
    offline_outputAC = [offline_outputAC outputAC];

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
    offline_outputAC = [offline_outputAC outputAC];

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
% save('offline_outputAC.mat','offline_outputAC')
figure(1)
plot(1:n,offline_outputAC)

