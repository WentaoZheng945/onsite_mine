%load("a_brake.mat");
%load("a_thr.mat");
%load("brake.mat");
%load("thr.mat");
%
%v0=0;
%steer=0;
%head=0;
%acceleration=1.2;
%gear=1;
%x0=0;
%y0=0;

function init(x, y, yaw, v0, acc)
    load("a_brake.mat");
    load("a_thr.mat");
    load("brake.mat");
    load("thr.mat");

    % x; 使用传入的参数值 (m)
    % y; % 使用传入的参数 (m)
    % yaw; % 使用传入的参数 (弧度or角度？, 东偏北方向?, 范围是多少？)
    % v0; % 使用传入的参数 (m/s)
    % acc; % 使用传入的参数 (m/s2)
    gear = 1;  % 初始化为1
    steer=0;  % 前轮转角，初始化为0
end



