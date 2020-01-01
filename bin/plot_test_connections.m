clear all; close all;
data=load('test_connections.dat');
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
plot(t,n1>-45 | n2>-45 | n3>-45 | n4>-45)
hold on
plot(t,(y1+70.6)*70,'r')
%
figure(7)
plot(t,n5>-45)
hold on
plot(t,(y2+70.6)*70,'r')
%
figure(8)
plot(t,n6>-45 | n7>-45)
hold on
plot(t,(y3+70.6)*70,'r')
