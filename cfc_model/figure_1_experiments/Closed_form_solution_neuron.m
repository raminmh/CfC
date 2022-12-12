function [v_final]=Closed_form_solution_neuron(neuron_number,v_pre,delta_t,y_init,g_l,v_l,c_m,sigma,mu,W,E)

y_1 = y_init;
%[v_pre_resampled time_ref]= resample(v_pre,[1:size(v_pre,1)],1/delta_t);
time = 0:delta_t:(size(v_pre{neuron_number},1)*delta_t)-delta_t;
time = time';
v_connections_pre = zeros(size(v_pre{neuron_number},1),1);
%tau = (y_1 - sum(E)).*exp(-time'.*[(g_l(1,1)./c_m(1,1)) + sum([W.*1./(exp(-sigma.*(v_pre{neuron_number}'-mu))+1)],1)./c_m(1,1)]);
%v_final = tau.*sum(1./(exp(sigma.*(v_pre{neuron_number}'-mu))+1).^W,1)+sum(v_l(1,1)+E);
for i=1:size(v_pre{neuron_number},2)
     
     v_connections(:,i) = (y_1 - E(i,1)).* exp(-time.*[(g_l(1,1)./c_m(1,1)) + [W(i)*1./(exp(-sigma(i,1).*(v_pre{neuron_number}(:,i)-mu(i,1)))+1)]/c_m(1,1)]).*(1./(exp(sigma(i,1).*(v_pre{neuron_number}(:,i)-mu(i,1)))+1).^W(i))+E(i);
     %v_connections(:,i) = (y_1 - E(i,1)).* exp(-time.*[(g_l(1,1)./c_m(1,1))]).*(1./(exp(sigma(i,1).*(v_pre{neuron_number}(:,i)-mu(i,1)))+1).^W(i))+E(i);
     v_final = v_connections_pre + v_connections(:,i);
     v_connections_pre = v_final;   
end
%v_final = v_final + v_l(1,1);


clear g_l v_l c_m sigma  mu W E time v_connections
