%Compute model performance statistics for RMA2 model output

clear


%load the data into Matlab

model = load('../../raw_results/calibration/model_n0.015_ev9000.txt');   %<<<< edit filename as required

observed = load('fowey_observed_data.txt');   %observed data


%pull out the columns and put them in sensibly named vectors

hours = model(:,1);
velMod = model(:,2); %modelled velocity magnitude
velObs = observed(:,2); %observed velocity magnitude
%Water level
levelMod = model(:,4); %modelled level
levelObs = observed(:,4); %observed level

%1: first step is always to visualise the series

figure(1)
 subplot(2,1,1)
  plot(hours,levelMod,'r',hours,levelObs, 'b') %model = red, obs = blue
  xlabel('Hours'); ylabel('Level (m OD)');
 subplot(2,1,2)
  plot(hours,velMod,'r',hours,velObs, 'b');
  xlabel('Hours'); ylabel('Speed (m/s)');
%2: second step is to compare observed versus modelled values

figure(2)
 subplot(1,2,1)
  plot(levelMod,levelObs,'+');
  xlabel('Model level (m OD)'); ylabel('Observed level (m OD)');
  
  hold on
  parityX = [-1.5 1.5];
  parityY = [-1.5 1.5];
  plot(parityX,parityY,'r');
  hold off
  
 subplot(1,2,2)
  plot(velMod,velObs,'+');
  xlabel('Model speed (m/s)'); ylabel('Observed speed (m/s)');
  
  hold on
  parityX = [0 1];
  parityY = [0 1];
  plot(parityX,parityY,'r');
  hold off


%3: compute objective performance functions
%NB: we should be cautious of including the initial part of the model
%output series since this may be affected by the 'spin-up' from a cold
%start condition. A sensible move in this case would be to discard the
%first tidal cycle - i.e. the first 12 hours or 48 readings (timestep is
%0.25 hr)

levelObs = levelObs(49:297);
levelMod = levelMod(49:297);
velObs = velObs(49:297);
velMod = velMod(49:297);
N = length(levelObs);

%RMS error
RMSE_level = sqrt(sum((levelObs-levelMod).^2) / N) %units are m
RMSE_velmag = sqrt(sum((velObs-velMod).^2) / N) %units are m/s

%NSE

NSE_level = 1 - ((sum((levelObs - levelMod) .^ 2) ) / (sum((levelObs - mean(levelObs)) .^2)))

NSE_vel = 1 - ((sum((velObs - velMod) .^ 2) ) / (sum((velObs - mean(velObs)) .^2)))
