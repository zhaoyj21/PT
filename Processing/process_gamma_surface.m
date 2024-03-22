global ev2j fontsize fontname
fontsize=8;
fontname='Arial';
ev2j=1.602176565e-19;

path_gsf='C:/Users/dell/Desktop/gsf/';
mat='cu';
lattice='fcc';
file='G01.data';


if exist([path_gsf,file])
    [lx,ly,natom,xn,yn,xstep,ystep,xdisp,ydisp,energy]=read_data(path_gsf,file);
    e=zeros(xn+1,yn+1);
    x=zeros(xn+1,yn+1);
    y=zeros(xn+1,yn+1);            
    for k=1:size(xstep,1)
        e(xstep(k)+1,ystep(k)+1)=energy(k)/lx/ly*1000;
        x(xstep(k)+1,ystep(k)+1)=xdisp(k)+1;
        y(xstep(k)+1,ystep(k)+1)=ydisp(k)+1;
    end
    [X,Y]=meshgrid([0:xn],[0:yn]);
    e=e-e(1,1);
    if strcmp(lattice,'fcc')
        x1=xn*2/3;
        y1=yn*2/3;
        sf=interp2(X,Y,e,x1,y1);
        x2=X(xn+1,1);
        y2=Y(xn+1,1);
        k=(y2-y1)/(x2-x1);
        b=(x2*y1-x1*y2)/(x2-x1);
        xq=linspace(x2,x1);
        yq=k*xq+b;
        usf_line=interp2(X,Y,e,xq,yq);
        usf=max(usf_line);                
    elseif strcmp(lattice,'bcc')
        sf=e(end,1);
        usf=max(e(:,1));                
    end
end

function [lx,ly,natom,xn,yn,xstep,ystep,xdisp,ydisp,energy]=read_data(path,file)
    global ev2j
    fileID=fopen([path,file]);
    tmp=cell2mat(textscan(fileID,'%f %f %f %f %f','HeaderLines',1));
    lx=tmp(1)*1e-10;
    ly=tmp(2)*1e-10;
    natom=tmp(3);
    xn=tmp(4);
    yn=tmp(5);
    tmp=cell2mat(textscan(fileID,'%f %f %f %f %f','HeaderLines',1));
    xstep=tmp(:,1);
    ystep=tmp(:,2);
    xdisp=tmp(:,3)*1e-10;
    ydisp=tmp(:,4)*1e-10;
    energy=tmp(:,5)*ev2j;
    fclose(fileID);
end