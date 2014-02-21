%Visualise changes to due to dredging in tidal estuary

clear


%load the data into Matlab

model = load('dredging_effects.txt');   %<<<< edit filename as required

%pull out the columns and put them in sensibly named vectors

hours = model(:,1); %we only need create one time column

reference_level = model(:,8); %this should be level at the mouth, which
%is useful to plot simply as a reference plot against which judge changes
%in other variables

%velocity changes
UpperVel = model(:,10); %Upper estuary change in velocity magnitude
MidVel = model(:,12); %Mid-estuary estuary change in velocity magnitude
BerthVel = model(:,14); %Berth estuary change in velocity magnitude
InletVel = model(:,16); %Inlet estuary change in velocity magnitude

%water level changes
UpperVel = model(:,18); %Upper estuary change in water level
MidVel = model(:,20); %Mid-estuary estuary change in water level
BerthVel = model(:,22); %Berth estuary change in water level
InletVel = model(:,24); %Inlet estuary change in water level

%Visualise the combination of series we want - subplots work well here

figure(3)
subplot(2,1,1)
 plot(hours,reference_level)
 xlabel('Hours')
 ylabel('Inlet water level')
subplot(2,1,2)
 plot(hours,UpperVel,'k',hours,MidVel,'r',hours,BerthVel,'b',hours,InletVel,'g');
 xlabel('Hours')
 ylabel('Delta velocity m/s')

 
%You can edit this as required to vary the plots and also plot level
%changes instead of velocity changes


