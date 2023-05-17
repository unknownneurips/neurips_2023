clear

Aeq = readNPY('Aeq.npy');
beq = readNPY('beq.npy'); 
A = readNPY('A.npy');
b = readNPY('b.npy');
lb = readNPY('lb.npy');
ub = readNPY('ub.npy');
intcon = readNPY('intcon.npy');  
intcon = double(intcon+1);
f = readNPY('f.npy'); 

x = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub);

vp = x(length(x)-1)

