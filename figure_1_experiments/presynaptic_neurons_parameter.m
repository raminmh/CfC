function [synapse_values,neuron_values,v_pre]= presynaptic_neurons_parameter(neuron_num,synapse_param_inter,synapse_param_sensory,neuron_param,neurons,sensory)

% select presynaptic neurons and their parameter for a particular neuron:
synapse_param_inter = [synapse_param_inter;synapse_param_sensory];
synapse_param_inter(35:end,1) = synapse_param_inter(35:end,1) + 19;
neurons = [neurons,sensory];
neuron_param = [neuron_param;zeros(size(sensory,2),3)];
for i= 1:size(synapse_param_inter,1)
    if synapse_param_inter(i,2) == neuron_num
        synapse_param{i,1} = synapse_param_inter(i,3:6);
    end
end

synapse_param_new = synapse_param(~cellfun(@isempty, synapse_param));
synapse_param_indexes = find(~cellfun(@isempty,synapse_param));

value = synapse_param_new;
index = synapse_param_indexes;

neuron_values{1} = neuron_param(neuron_num+1,:);
for i= 1:(size(index,1))
  x = synapse_param_inter(index(i),1);
  v_pre{i} = neurons(:,x+1);
  neuron_values{i+1,1} = neuron_param(x+1,:);
end

v_pre = cell2mat(v_pre);
synapse_values = cell2mat(value);
neuron_values = cell2mat(neuron_values);


