function [y_downsampled,tau]=neuron(v_pre,delta_t,y_init,g_l,v_l,c_m,sigma,mu,W,E)

y_1 = y_init;
%[v_pre_resampled time_ref]= resample(v_pre,[1:size(v_pre,1)],1/delta_t);
for i = 1:size(v_pre,1)

dydt = (-(g_l + sum(W.*1./(1+ exp(-sigma'.*(v_pre(i,:)-mu')))',1)').*y_1 + g_l.*v_l + sum(W.*1./(1+ exp(-sigma'.*(v_pre(i,:)-mu')))'.*E,1)')./c_m;
y(i,1) = y_1 + delta_t * dydt;

y_1 = y(i,1);

tau(i,1) =1./((g_l + sum(W.*1./(1+ exp(-sigma'.*(v_pre(i,:)-mu')))',1)')./c_m);

end

y_downsampled = resample(y,[1:size(v_pre,1)],1/1);
%figure;plot(y)
clear g_l v_l c_m sigma  mu W E
end

