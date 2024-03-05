
    %% 每一次进行新仿真前运行清理资源
    % fclose(u);
    % delete(u);
    % clear u;

    %% 启动仿真
    % clear,clc,close all;

    %加载初始状态
    %run("init.m");
    %x=0;
    %y=0;
    %yaw=0;
    %v0=0;
    %acc=1.2;
    %init(x, y, yaw, v0, acc); % 直接调用init函数
    % x; 使用传入的参数值 (m)
    % y; % 使用传入的参数 (m)
    % yaw; % 使用传入的参数 (弧度or角度？, 东偏北方向?, 范围是多少？)
    % v0; % 使用传入的参数 (m/s)
    % acc; % 使用传入的参数 (m/s2)

    % 设置UDP参数
    localPort = 25001; % MATLAB监听的端口
    remotePort = 25000; % 发送指令的远程端口
    remoteIP = '127.0.0.1'; % 远程IP地址

    % 创建UDP对象
    u = udp(remoteIP, 'LocalPort', localPort, 'RemotePort', remotePort);
    fopen(u);
    modelName = 'VehicleModel'; % 仿真模型的名字
    load_system(modelName);
    set_param(modelName, 'StopTime', '150');
    
    % 发送准备就绪的信号
    readyMessage = 'ready';
    fwrite(u, readyMessage, 'char'); % 将'ready'字符串作为字符数组发送


    disp("开始接受消息")
    if_first =true;
    try
        while true
            % 检查是否收到UDP消息
            if u.BytesAvailable > 0
                disp('收到消息')
                % 读取数据（此处不做处理，仅确认收到消息）
                data=fread(u, u.BytesAvailable,'double');
                
                gear = data(1);
                acc = data(2);
                yaw = data(3);

                set_param(modelName,'SimulationCommand','update');
                
                % 获取当前仿真时间
                %simTime = get_param(modelName, 'SimulationTime');

                % 设置仿真的下一步停止时间

                % set_param(modelName,'SimulationCommand','')
                % 启动或继续仿真
                if if_first == true
                    set_param(modelName, 'SimulationCommand', 'start');
                    if_first = false;
                else
                    set_param(modelName, 'SimulationCommand', 'step');
                end
                while true
                    if strcmp(get_param(modelName,'SimulationStatus'),'paused')
                        break;
                    else
                        pause(0.1);
                    end
                end
                current_velocity = out.velocity.Data(end);
                current_head = out.phi.Data(end);
                current_x = out.x.Data(end);
                current_y = out.y.Data(end);
                

                % 等待仿真达到指定时间
                % while true
                %     stoptime=get_param(modelName, 'SimulationTime');
                %     if stoptime > (simTime + simDuration)
                %         disp(stoptime-(simTime + simDuration));
                %         break;
                %     else
                %         pause(0.0013); % 短暂暂停以减少CPU占用
                %     end
                %end

                % 暂停仿真等待下一条消息
                %set_param(modelName, 'SimulationCommand', 'pause');
                %disp(out.yout{1}.Values.Data);
                % listener = add_exec_event_listener(modelName, 'PostOutputs', @myPauseListener);
                %current_head = oget_param(modelName,'Outport2');
                %current_x = get_param(modelName,'Outport2');
                %current_y = get_param(modelName,'Outport2');

                fwrite(u,[current_velocity,current_head,current_x,current_y],'double');
                %disp([current_velocity,current_head,current_x,current_y]);
            else
                pause(0.05); % 没有消息时短暂暂停减少CPU占用
            end
        end
    catch ME
        disp(['Error: ', ME.message]);
    end

    % 清理资源
    fclose(u);
    delete(u);
    clear u;

    % function myPauseListener()
    %     % 获取并处理指定block的输出数据
    %     outData = get_param(modelName+"/current_velocity", 'ObjectParameters');
    %     % 这里可以进行数据处理，例如打印或保存数据
    %     disp(['Output at pause: ', num2str(outData)]);
    % end
