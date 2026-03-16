clc;
clear;

disp('Enter weights');
w1 = input('Weight w1 = ');
w2 = input('Weight w2 = ');

disp('Enter Threshold Value');
theta = input('theta = ');

y = [0 0 0 0];
x1 = [0 0 1 1];
x2 = [0 1 0 1];
z  = [0 0 1 0];   % Desired output for ANDNOT

con = 1;

while con == 1
    
    zin = x1*w1 + x2*w2;
    
    for i = 1:4
        if zin(i) >= theta
            y(i) = 1;
        else
            y(i) = 0;
        end
    end
    
    disp('Output of Net');
    disp(y);
    
    if y == z
        con = 0;
    else
        disp('Net is not learning. Enter another set of weights and threshold value');
        w1 = input('Weight w1 = ');
        w2 = input('Weight w2 = ');
        theta = input('theta = ');
    end

end

disp('McCulloch-Pitts Net for ANDNOT function');
disp('Weights of Neuron');
disp(w1);
disp(w2);
disp('Threshold value');
disp(theta);