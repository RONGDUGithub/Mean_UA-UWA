t=-1:0.01:1;

epsilon=0.5;

eexp=exp(epsilon);
eexp2=exp(epsilon/2);
exp3=exp(-epsilon/2);
epsilonstar=0.61;

if epsilon>epsilonstar
    alpha=1-exp3;
else
     alpha=0;
end

budget=eexp;
b=(epsilon*budget-budget+1)/(2*budget*(budget-1-epsilon));
%high_area  S_h
p=budget/(2*b*budget+1);
%low_area  S_l
q=1/(2*b*budget+1);


Var_sr=power((eexp+1)/(eexp-1),2)-power(t,2);
Var_pm=power(t,2)/(eexp2-1)+(eexp2+3)/(3*power((eexp2-1),2));
Var_hm=alpha*Var_pm+(1-alpha)*Var_sr;

Var_laplace=8/power(epsilon,2);
t2=(t+1)/2;
Var_sw=4*(q*((1+3*b+3*power(b,2)-6*b*power(t2,2))/3)+p*((6*b*power(t2,2)+2*b^3)/3)-power(t2*2*b*(p-q)+q*(b+1/2),2));
Var_sw_unbiase=Var_sw/(power(2*b*(p-q),2));


figure;

plot(t, Var_sr, 'LineWidth', 2, 'Color', 'blue');
hold on;
plot(t, Var_pm, '--', 'LineWidth', 2, 'Color', 'red');
plot(t, Var_hm, ':', 'LineWidth', 2, 'Color', 'green');
plot(t, Var_laplace, '-.', 'LineWidth', 2, 'Color', 'magenta');
plot(t, Var_sw, 'LineWidth', 2, 'Color', 'cyan');
plot(t, Var_sw_unbiase, '--', 'LineWidth', 2, 'Color', 'black');

legend('SR', 'PM', 'HM', 'Laplace','SW', 'SW_{unbiase}');