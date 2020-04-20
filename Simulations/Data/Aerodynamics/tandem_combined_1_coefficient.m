%Combined results for upstream model constant, downstream model varying

clear all
close all
clc
%% load in data
load('results_smooth1');
results_smooth1=results_smooth;
clear('results_smooth')
load('results_smooth2');
results_smooth2=results_smooth;
clear('results_smooth')

smooth1=(results_smooth1+results_smooth2)/2;

load('tlift.mat')
load('tdrag.mat')
load('tliftr.mat')
load('tdragr.mat')

colorset=[.6 0 0; .8 0.48 .38; 0.9 0.7 0.7; .85    .82    .82];

liftc=(tlift(:,:,1:4)+tliftr(:,:,5:8))/2;
dragc=(tdrag(:,:,1:4)+tdragr(:,:,5:8))/2;
lodc=liftc./dragc;

index=[3 7 11 15];

%single model comparison values
for i=1:1:4
    lcomp(i)=(smooth1(index(i),21)+smooth1(9,21))/2;
    dcomp(i)=(smooth1(index(i),22)+smooth1(9,22))/2;
    lodcomp(i)=lcomp(i)/dcomp(i);
end

%% plot figures
figure1=figure('name','Change in Lift','color',[1 1 1],'position',[50 50 1000 700]);
xlimit=[0 2];
subplot(221)
set(gca,'ColorOrder',colorset);
hold all
i=1;
plot(liftc(:,4,1),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(liftc(:,3,1),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(liftc(:,2,1),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(liftc(:,1,1),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_L (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_L, Downstream AoA = 0\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(222)
i=2;
set(gca,'ColorOrder',colorset);
hold all
plot(liftc(:,4,2),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(liftc(:,3,2),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(liftc(:,2,2),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(liftc(:,1,2),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_L (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_L, Downstream AoA = 20\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(223)
i=3;
set(gca,'ColorOrder',colorset);
hold all
plot(liftc(:,4,3),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(liftc(:,3,3),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(liftc(:,2,3),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(liftc(:,1,3),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_L (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_L, Downstream AoA = 40\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(224)
i=4;
set(gca,'ColorOrder',colorset);
hold all
plot(liftc(:,4,4),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(liftc(:,3,4),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(liftc(:,2,4),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(liftc(:,1,4),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_L (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_L, Downstream AoA = 60\circ')
legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
xlim(xlimit)

%% plot of change in drag

figure2=figure('name','Change in Drag','color',[1 1 1],'position',[50 50 1000 700]);
xlimit=[0 1.5];
subplot(221)
i=1;
set(gca,'ColorOrder',colorset);
hold all
plot(dragc(:,4,1),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(dragc(:,3,1),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(dragc(:,2,1),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(dragc(:,1,1),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_D (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_D, Downstream AoA = 0\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(222)
i=2;
set(gca,'ColorOrder',colorset);
hold all
plot(dragc(:,4,2),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(dragc(:,3,2),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(dragc(:,2,2),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(dragc(:,1,2),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_D (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_D, Downstream AoA = 20\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(223)
i=3;
set(gca,'ColorOrder',colorset);
hold all
plot(dragc(:,4,3),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(dragc(:,3,3),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(dragc(:,2,3),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(dragc(:,1,3),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_D (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_D, Downstream AoA = 40\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(224)
i=4;
set(gca,'ColorOrder',colorset);
hold all
plot(dragc(:,4,4),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(dragc(:,3,4),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(dragc(:,2,4),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(dragc(:,1,4),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta C_D (%)')
ylabel('Vertical Spacing, chords')
title('Combined C_D, Downstream AoA = 60\circ')
legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
xlim(xlimit)


%% plot of change in lod

figure3=figure('name','Change in LoD','color',[1 1 1],'position',[50 50 1000 700]);
xlimit=[0 2.6];
subplot(221)
i=1;
set(gca,'ColorOrder',colorset);
hold all
plot(lodc(:,4,1),[0 1 2 3 4 5],'-d','LineWidth',2,'MarkerFaceColor',colorset(1,:))
plot(lodc(:,3,1),[0 1 2 3 4 5],'-s','LineWidth',2,'MarkerFaceColor',colorset(2,:))
plot(lodc(:,2,1),[0 1 2 3 4 5],'-^','LineWidth',2,'MarkerFaceColor',colorset(3,:))
plot(lodc(:,1,1),[0 1 2 3 4 5],'-o','LineWidth',2,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta L/D (%)')
ylabel('Vertical Spacing, chords')
title('Combined L/D, Downstream AoA = 0\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(222)
i=2;
set(gca,'ColorOrder',colorset);
hold all
plot(lodc(:,4,2),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(lodc(:,3,2),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(lodc(:,2,2),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(lodc(:,1,2),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta L/D (%)')
ylabel('Vertical Spacing, chords')
title('Combined L/D, Downstream AoA = 20\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(223)
i=3;
set(gca,'ColorOrder',colorset);
hold all
plot(lodc(:,4,3),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(lodc(:,3,3),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(lodc(:,2,3),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(lodc(:,1,3),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta L/D (%)')
ylabel('Vertical Spacing, chords')
title('Combined L/D, Downstream AoA = 40\circ')
xlim(xlimit)
%legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')

subplot(224)
i=4;
set(gca,'ColorOrder',colorset);
hold all
plot(lodc(:,4,4),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(lodc(:,3,4),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(lodc(:,2,4),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(lodc(:,1,4),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta L/D (%)')
ylabel('Vertical Spacing, chords')
title('Combined L/D, Downstream AoA = 60\circ')
legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
xlim(xlimit)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    ANOVA Analysis    %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AoA_vec=[0*ones(1,24),20*ones(1,24),40*ones(1,24),60*ones(1,24)];

Vert=repmat([0 1 2 3 4 5],[1,16]);

Hz=[8*ones(1,6),6*ones(1,6),4*ones(1,6),2*ones(1,6)];
Hz=repmat(Hz,[1 4]);

data=[Hz',Vert',AoA_vec'*pi/180];

[p_lod,table,stats,terms]=anovan(lodc(:),data,'model','interaction');

a0=.2*ones(1,19);
[a,resnorm,residual]=lsqcurvefit(@myfun,a0,data',lodc(:)');

figure('color',[1 1 1],'position',[50 50 1000 700])
i=2;
set(gca,'ColorOrder',colorset);
hold all
plot(lodc(:,4,2),[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(lodc(:,3,2),[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(lodc(:,2,2),[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(lodc(:,1,2),[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
grid on
xlabel('\Delta L/D (%)')
ylabel('Vertical Spacing, chords')
title('Combined L/D, Downstream AoA = 20\circ')
xlim(xlimit)

hh=[8 6 4 2];
vv=[0 1 2 3 4 5];
aa=[20]*pi/180;

for i=1:4
    for j=1:6        
        lodfit(j,i)=feval(@myfun,a,[hh(i),vv(j),aa]');
    end
end

plot(lodfit(:,4),[0 1 2 3 4 5],'--d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
plot(lodfit(:,3),[0 1 2 3 4 5],'--s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
plot(lodfit(:,2),[0 1 2 3 4 5],'--^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
plot(lodfit(:,1),[0 1 2 3 4 5],'--o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% colorset=flipud(colorset);
% figure4=figure('color',[1 1 1],'name','Combined Lift','position',[50 50 1000 400]);
% xlimit=[-90 90];
% xtick=[-90 -60 -30 0 30 60 90];
% subplot(141)
% set(gca,'ColorOrder',colorset);
% hold all
% i=1;
% plot((liftc(:,4,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% i=2;
% plot((liftc(:,4,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% i=3;
% plot((liftc(:,4,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% i=4;
% plot((liftc(:,4,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta C_L (%)')
% %ylabel('Vertical Spacing, chords')
% title('Combined C_L, H = 2 Chords')
% ylim([0 5.05])
% set(gca,'YTick',[0 1 2 3 4 5],'XTick',xtick)
% %legend('AoA = 0\circ','AoA = 20\circ','AoA = 40\circ','AoA = 60\circ')
% xlim(xlimit)
% 
% subplot(142)
% set(gca,'ColorOrder',colorset);
% hold all
% i=1;
% plot((liftc(:,3,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% i=2;
% plot((liftc(:,3,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% i=3;
% plot((liftc(:,3,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% i=4;
% plot((liftc(:,3,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta C_L (%)')
% %ylabel('Vertical Spacing, chords')
% title('Combined C_L, H = 4 Chords')
% ylim([0 5.05])
% set(gca,'YTick',[0 1 2 3 4 5],'XTick',xtick)
% %legend('AoA = 0\circ','AoA = 20\circ','AoA = 40\circ','AoA = 60\circ')
% xlim(xlimit)
% 
% subplot(143)
% set(gca,'ColorOrder',colorset);
% hold all
% i=1;
% plot((liftc(:,2,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% i=2;
% plot((liftc(:,2,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% i=3;
% plot((liftc(:,2,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% i=4;
% plot((liftc(:,2,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta C_L (%)')
% %ylabel('Vertical Spacing, chords')
% title('Combined C_L, H = 6 Chords')
% ylim([0 5.05])
% set(gca,'YTick',[0 1 2 3 4 5],'XTick',xtick)
% %legend('AoA = 0\circ','AoA = 20\circ','AoA = 40\circ','AoA = 60\circ')
% xlim(xlimit)
% 
% subplot(144)
% set(gca,'ColorOrder',colorset);
% hold all
% i=1;
% plot((liftc(:,1,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% i=2;
% plot((liftc(:,1,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% i=3;
% plot((liftc(:,1,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% i=4;
% plot((liftc(:,1,i)-lcomp(i))/lcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% set(gca,'YTick',[0 1 2 3 4 5],'XTick',[-60 -30 0 30 60])
% xlabel('\Delta C_L (%)')
% %ylabel('Vertical Spacing, chords')
% title('Combined C_L, H = 8 Chords')
% ylim([0 5.05])
% legend('AoA = 0\circ','AoA = 20\circ','AoA = 40\circ','AoA = 60\circ')
% xlim(xlimit)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% figure4=figure('name','Change in LoD','color',[1 1 1],'position',[50 50 1000 700]);
% xlimit=[-40 160];
% subplot(2,3,1:2)
% i=1;
% set(gca,'ColorOrder',colorset);
% hold all
% plot((lodc(:,4,1)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2,'MarkerFaceColor',colorset(1,:))
% plot((lodc(:,3,1)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2,'MarkerFaceColor',colorset(2,:))
% plot((lodc(:,2,1)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2,'MarkerFaceColor',colorset(3,:))
% plot((lodc(:,1,1)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta L/D (%)')
% ylabel('Vertical Spacing, chords')
% title('Combined L/D, Downstream AoA = 0\circ')
% xlim(xlimit)
% %legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
% 
% subplot(233)
% i=2;
% set(gca,'ColorOrder',colorset);
% hold all
% plot((lodc(:,4,2)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% plot((lodc(:,3,2)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% plot((lodc(:,2,2)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% plot((lodc(:,1,2)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta L/D (%)')
% ylabel('Vertical Spacing, chords')
% title('Combined L/D, Downstream AoA = 20\circ')
% xlim([-40 40])
% %legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
% 
% subplot(234)
% i=3;
% set(gca,'ColorOrder',colorset);
% hold all
% plot((lodc(:,4,3)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% plot((lodc(:,3,3)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% plot((lodc(:,2,3)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% plot((lodc(:,1,3)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta L/D (%)')
% ylabel('Vertical Spacing, chords')
% title('Combined L/D, Downstream AoA = 40\circ')
% xlim([-40 40])
% %legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
% 
% subplot(2,3,5:6)
% i=4;
% set(gca,'ColorOrder',colorset);
% hold all
% plot((lodc(:,4,4)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-d','LineWidth',2.5,'MarkerFaceColor',colorset(1,:))
% plot((lodc(:,3,4)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-s','LineWidth',2.5,'MarkerFaceColor',colorset(2,:))
% plot((lodc(:,2,4)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-^','LineWidth',2.5,'MarkerFaceColor',colorset(3,:))
% plot((lodc(:,1,4)-lodcomp(i))/lodcomp(i)*100,[0 1 2 3 4 5],'-o','LineWidth',2.5,'MarkerFaceColor',colorset(4,:))
% grid on
% xlabel('\Delta L/D (%)')
% ylabel('Vertical Spacing, chords')
% title('Combined L/D, Downstream AoA = 60\circ')
% legend('Horz Spacing = 2c','Horz Spacing = 4c','Horz Spacing = 6c','Horz Spacing = 8c')
% xlim(xlimit)
