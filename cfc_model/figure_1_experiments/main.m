

%% load neural activity of the active road test:

load('figure_1_data.mat');
experiment = neurons1;
experiment_sensors = sensory1;
experiment_output = output1;


%% get the presynaptic parameters for all LTC neurons and reproduce their dynamics:
delta_t = 0.001;

%figure
for i=1:size(experiment,2)
  [s_values{i},n_values{i},v_pre{i}] = presynaptic_neurons_parameter(i-1,synapse_param_inter,synapse_param_sensory,neuron_param,experiment,experiment_sensors);
switch_param = i;
  switch i
      case 7
        delta_t = 0.0001;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4)); 
      case 5
        delta_t = 1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4)); 
      case 6
        delta_t = 1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4)); 
      case 2
        delta_t = 1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4)); 
      case 16
        delta_t = 1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 3
        delta_t = 2;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4)); 
      case 10
        delta_t = 0.9;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4)); 
      case 19
        delta_t = 0.8;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 18
        delta_t = 0.7;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 9
        delta_t = 0.6;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 4
        delta_t = 0.5;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 13
        delta_t = 0.5;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 14
        delta_t = 0.5;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      %case 5
       % delta_t = 0.3;
       % [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 1
        delta_t = 0.1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 15
        delta_t = 0.1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,0,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 8
        delta_t = 1;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,-1.43,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 12
        delta_t = 0.2;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,-0.7246,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 17
        delta_t = 0.044;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,-0.5,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));  
      case 11
        delta_t = 0.02;
        [n_out{i},tau{i}] = neuron(v_pre{i},delta_t,-0.5,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));   
  end
%   subplot(5,4,i)
%    plot(n_out{i})
%    hold on 
%    plot(experiment(:,i))
%    hold off
   %clear s_values{i} n_values{i}
end
%    plot(n_out{11})
%    xlim([3400,4400])
%    hold on 
%    plot(neurons(:,11))
%    xlim([3400,4400])
%    hold off

% predictions = cell2mat(n_out);
% 
% number_of_neurons = size(experiment,2);
% figure
% ccc =linspace(1,10,size(experiment,1));
% for i= 1:number_of_neurons
% subplot(4,5,i)
% scatter(tau{i},experiment(:,i),10,ccc, 'filled')
% %yticks([]);
% %xticks([]);   
% % xlim([-72 -62])
% % xlim([-70.1 -69.5]) %for neuron 8
% % xlim([-70 -17]) %for neuron 10
% % ylim([-60 -30])
% %colorbar;
% end

%% Compute the closed-form solution and plot it together with the actual LTC output
delta_t = 0.00001;
neuron_number = 1;
i=1;
closed_form_out2 = Closed_form_solution_neuron(neuron_number,v_pre,delta_t,3,n_values{i}(1,1),n_values{i}(1,2),n_values{i}(1,3),s_values{i}(:,1),s_values{i}(:,2),s_values{i}(:,3),s_values{i}(:,4));
figure
plot(neurons1(:,1))
hold on
plot(closed_form_out2)
hold off
axis('off')