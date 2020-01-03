clear all; close all;
data=load('test_connections_voltage.dat');
t=data(:,1);
n1=data(:,2);
n2=data(:,3);
n3=data(:,4);
n4=data(:,5);
n5=data(:,6);
n6=data(:,7);
n7=data(:,8);
%
y1=data(:,9);
y2=data(:,10);
y3=data(:,11);
%
data=load('test_connections_g1.dat');
z1=data(:,2);
z2=data(:,3);
z3=data(:,4);
z4=data(:,5);
z5=data(:,6);
z6=data(:,7);
%
%
data=load('test_connections_spikes.dat');
sp1=data(:,2);
sp2=data(:,3);
sp3=data(:,4);
sp4=data(:,5);
sp5=data(:,6);
sp6=data(:,7);
sp7=data(:,8);

figure(1);
plot(t,n1);
figure(2);
plot(t,n7)
%
figure(3);
plot(t,y1)
%
figure(4);
plot(t,y2)
%
figure(5);
plot(t,y3)
%
figure(6)
plot(t,sp1+sp2+sp3+sp4)
hold on
plot(t,(y1+70.6)*70,'r')
%
figure(7)
plot(t,sp5)
hold on
plot(t,(y2+70.6)*70,'r')
%
figure(8)
plot(t,sp6 + sp7)
hold on
plot(t,(y3+70.6)*70,'r')
%
figure(9)
plot(t,sp1)
hold on
plot(t,z1*50,'r')
%
figure(10)
plot(t,sp2+sp3)
hold on
plot(t,z2*50,'r')
%
figure(11)
plot(t,sp4)
hold on
plot(t,z3*50,'r')
%
figure(12)
plot(t,sp5)
hold on
plot(t,z4*50,'r')
%
figure(13)
plot(t,sp6)
hold on
plot(t,z5*50,'r')
%
figure(14)
plot(t,sp7)
hold on
plot(t,z6*50,'r');
%
