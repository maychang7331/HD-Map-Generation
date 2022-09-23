clc;
clear all;
% 使用 "RPY"

%% test
% Xc = 177012.1852;
% Yc = 2535867.5108;
% Zc = 46.6994;

% RPY (最後是用這個！！！)
% roll = -81.72804173899999;
% pitch = -4.606952759;
% yaw = -164.57782739299998;

% OPK
% roll = 114.053360293
% pitch = -71.560626312
% yaw = 28.423823772

%% Cam1 015834051 左後
Xc = 177024.4084;
Yc = 2535875.9397;
Zc = 46.6897;

% RPY 
roll = -90.197685743;
pitch = 2.669087131;
yaw = 178.054815673;

% OPK
% roll = -89.802398413;
% pitch = 1.935967444;
% yaw = 177.324438952;

%% Cam2 015834051   右後
% Xc = 177025.2038;
% Yc = 2535876.2964;
% Zc = 46.6699;

% RPY 
% roll = -85.426284982;
% pitch = -0.992556225;
% yaw = -129.066255783;

% OPK
% roll = -97.245561134;
% pitch = -50.788315577;
% yaw = 175.369697727;

%% Cam3 015834051   左前
% Xc = 177024.2269;
% Yc = 2535876.2044;
% Zc = 46.7175;

% RPY 
% roll = -82.256296136;
% pitch = -4.568189481;
% yaw = 105.249235587;

% OPK
% roll = -116.367292567;
% pitch = 72.395980122;
% yaw = -150.100562313;

%% Cam4 015834052   右前
% Xc = 177025.2562;
% Yc = 2535876.7327;
% Zc = 46.5947;

% RPY 
% roll = -78.994272501;
% pitch = 1.836836002;
% yaw = -54.518422212;

% OPK
% roll = 108.667981763;
% pitch = -53.405640008;
% yaw = 13.305687843;


%%
Rx = [[1,0,0];[0,cosd(roll),sind(roll)*(-1)];[0,sind(roll),cosd(roll)]];
Ry = [[cosd(pitch),0,sind(pitch)];[0,1,0];[sind(pitch)*(-1),0,cosd(pitch)]];
Rz = [[cosd(yaw),sind(yaw)*(-1),0];[sind(yaw),cosd(yaw),0];[0,0,1]];

R = Rz*Ry*Rx;
T = [Xc; Yc; Zc];
RT = [R T; 0 0 0 1]
inv(RT)

U = R(1,1);
V = R(2,1);
W = R(3,1);
quiver3(Xc,Yc,Zc,U,V,W,'Color','b','LineWidth', 2, 'MaxHeadSize', 0.8);
hold on;

U = R(1,2);
V = R(2,2);
W = R(3,2);
quiver3(Xc,Yc,Zc,U,V,W,'Color','g','LineWidth', 2, 'MaxHeadSize', 0.8);
hold on;

U = R(1,3);
V = R(2,3);
W = R(3,3);
quiver3(Xc,Yc,Zc,U,V,W,'Color','r','LineWidth', 2, 'MaxHeadSize', 0.8);
hold on;

scatter3(177019.2065, 2535864.73, 47.2795);
axis equal
