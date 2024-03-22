%clear;
global ev2n fontsize fontname
fontsize=10;
fontname='Arial';
ev2n=1.6022E-19*1e10;   % metal unit in lammps

path='C:/Users/dell/Desktop/shear/';
file='G01';
fsuffix='.lammpstrj';
target=[1,4:400];   % the targeted frame in the trajetory files, the 1st one is the 
a=3.65;            % lattice constant
lattice='fcc';      % lattice type
ifprint=1;          % if print the figures or not
process_data(path,file,fsuffix,lattice,a,target,ifprint);
    
% group()  % % use to process a group of data     

function group()
    path0='F:\Data\MD\eam\shear\';
    mat={'w'};
    file='G01';
    fsuffix='.lammpstrj';
    target=[1,4:400];
    ifprint=1;
    lattice='bcc';
    load('D:\Tsinghua\Project\Fatigue of LJ materials\data\nist_repository_bcc.mat')
    nist_repo=data;
    clear data

    for i=1:numel(mat)
        tmp=dir([path0,mat{i}]);
        mname=mat{i};
        mname(1)=upper(mname(1));

        for j=3:numel(tmp)
            if ~(strcmp(tmp(j).name(1),'1') || strcmp(tmp(j).name(1),'2'))
                continue;
            end
            tic
            disp(tmp(j).name)
            path=[path0,mat{i},'\',tmp(j).name,'\'];
            id=get_dataset_id(getfield(nist_repo,mat{i}),tmp(j).name);
            eval(sprintf('a=nist_repo.%s(%d).a;',mat{i},id));  
            try
                process_data(path,file,fsuffix,lattice,a,target,ifprint);
            catch
                disp('ERROR')
            end
            toc
        end
    end
end

function process_data(path,file,fsuffix,lattice,a,target,ifprint)
    global ev2n fontsize fontname
    read_type='frame';
    ifrecursive=0;
    reload=1;
    
    if ~exist([path,file,'.mat']) || reload
        fid=fopen([path,file,fsuffix]);
        data=struct();
        [data.step,data.natom,data.id,data.coord,data.box,data.mapid]=read_data(fid,read_type,target,ifrecursive);
        %save([path,file,'.mat'],'data')
        fclose(fid);
    else
        load([path,file,'.mat'],'data')
    end

    [gamma,sxz,energy]=read_time_data(path,[file,'.data']);

    [topid,botid]=detect_dislocation_interface(data.coord{1});

    i=2;
    cutoff=a*2;
    tcrd0=data.coord{i}(topid,:);
    bcrd0=data.coord{i}(botid,:);
    if strcmp(lattice,'fcc')
        [neigh0,neigh02,bcrd0]=find_neigh_pbc(tcrd0,bcrd0,cutoff,data.box{i},[data.box{i}(2,2)-data.box{i}(2,1),data.box{i}(2,2)-data.box{i}(2,1)],3);   % find neigh for top atoms in the bottom atoms
        thre_dpos=a*sqrt(6)/24;%a*sqrt(6)/24;  % half of the radius of the incircle
    elseif strcmp(lattice,'bcc')
        [neigh0,neigh02,bcrd0]=find_neigh_pbc(tcrd0,bcrd0,cutoff,data.box{i},[data.box{i}(2,2)-data.box{i}(2,1),data.box{i}(2,2)-data.box{i}(2,1)],4);
        thre_dpos=a/4;  % half of the radius of the incircle
    end
        
    dpos=cal_dev_pos(tcrd0,bcrd0,neigh0);
    cand0=find(dpos<thre_dpos^2);
    flag=0;
    for i=3:numel(data.step)
        tcrd=data.coord{i}(topid,:);
        bcrd=data.coord{i}(botid,:);
        %if i==180
        %    plot(tcrd(:,1),tcrd(:,2),'b.')
        %    hold on
        %    plot(bcrd(:,1),bcrd(:,2),'r.')
        %    hold off
        %end
        if strcmp(lattice,'fcc')
            [neigh,neigh2,bcrd]=find_neigh_pbc(tcrd,bcrd,cutoff,data.box{i},[data.box{i}(2,2)-data.box{i}(2,1),data.box{i}(2,2)-data.box{i}(2,1)],3);   % find neigh for top atoms in the bottom atoms
            dpos=cal_dev_pos(tcrd,bcrd,neigh);
            cand=find(dpos<thre_dpos^2);
            nbchange=neigh_change(neigh02,neigh2,3);     
            slip=intersect(cand,nbchange);
            if ~isempty(slip) && sum(ismember(slip,cand0))
                flag=1;
                break;
            end            
        elseif strcmp(lattice,'bcc')
            [neigh,neigh2,bcrd]=find_neigh_pbc(tcrd,bcrd,cutoff,data.box{i},[data.box{i}(2,2)-data.box{i}(2,1),data.box{i}(2,2)-data.box{i}(2,1)],4);   % find neigh for top atoms in the bottom atoms
            dpos=cal_dev_pos(tcrd,bcrd,neigh);
            cand=find(dpos<thre_dpos^2);            
            nbchange=neigh_change(neigh02,neigh2,3);
            slip=intersect(cand,nbchange);
            if ~isempty(slip) && sum(ismember(slip,cand0))
                flag=1;
                break;
            end  
        end    
    end

    if flag
        disp(sxz(i-3))

        fid=fopen([path,'pn.data'],'w');
        fprintf(fid,'%f %f',thre_dpos,sxz(i-3));
        fclose(fid);

        id=2;
        fig1=figure(1);
        p1(1)=plot(tcrd(slip(id),1),tcrd(slip(id),2),'.b');
        hold on
        plot(bcrd(neigh(slip(id),:),1),bcrd(neigh(slip(id),:),2),'.r')
        xb(1)=min(bcrd(neigh(slip(id),:),1));
        xb(2)=max(bcrd(neigh(slip(id),:),1));
        yb(1)=min(bcrd(neigh(slip(id),:),2));
        yb(2)=max(bcrd(neigh(slip(id),:),2));

        p1(2)=plot(tcrd0(slip(id),1),tcrd0(slip(id),2),'ob');
        hold on
        plot(bcrd0(neigh0(slip(id),:),1),bcrd0(neigh0(slip(id),:),2),'or')
        xb(1)=min(min(bcrd0(neigh0(slip(id),:),1)),xb(1));
        xb(2)=max(max(bcrd0(neigh0(slip(id),:),1)),xb(2));
        yb(1)=min(min(bcrd0(neigh0(slip(id),:),2)),yb(1));
        yb(2)=max(max(bcrd0(neigh0(slip(id),:),2)),yb(2));

        len=max(xb(2)-xb(1),yb(2)-yb(1));
        xb(1)=xb(1)-len*0.2;
        xb(2)=xb(1)+len*1.4;
        yb(1)=yb(1)-len*0.2;
        yb(2)=yb(1)+len*1.4;
        xlim(xb)
        ylim(yb)
        xlabel('x / ang.')
        ylabel('y / ang.')
        legend(p1,{'origin','current'},'box','off')
        set(gcf,'unit','centimeters','position',[5,5,10,10]);
        set(gca,'unit','centimeters','position',[1.5,1.5,8,8]);
        set(gca,'FontSize',fontsize,'FontName',fontname,'LabelFontSizeMultiplier',1.0);
        hold off
        
        fig2=figure(2);
        plot(gamma(2:end),sxz(2:end),'b')
        hold on
        plot(gamma(i-3),sxz(i-3),'.r')
        xlabel('\gamma')
        ylabel('\sigma / MPa')
        title(num2str(sxz(i-3)))
        set(gcf,'unit','centimeters','position',[5,15,10,10]);
        set(gca,'unit','centimeters','position',[1.5,1.5,8,8]);
        set(gca,'FontSize',fontsize,'FontName',fontname,'LabelFontSizeMultiplier',1.0);
        hold off
        if ifprint
            if ~exist([path,'figure'])
                mkdir([path,'figure'])
            end
            print(fig1,[path,'figure\activated_atoms.jpg'],'-djpeg','-r300');
            print(fig2,[path,'figure\strain_stress.jpg'],'-djpeg','-r300');
        end
    else
        disp(['No transform, ',path])
        fid=fopen([path,'pn.data'],'w');
        fprintf(fid,'%f %f',thre_dpos,nan);
        fclose(fid);
        
        fig2=figure(2);
        plot(gamma(2:end),sxz(2:end),'b')
        xlabel('\gamma')
        ylabel('\sigma / MPa')
        title(num2str(sxz(i-3)))
        set(gcf,'unit','centimeters','position',[5,15,10,10]);
        set(gca,'unit','centimeters','position',[1.5,1.5,8,8]);
        set(gca,'FontSize',fontsize,'FontName',fontname,'LabelFontSizeMultiplier',1.0);  
        if ifprint
            if ~exist([path,'figure'])
                mkdir([path,'figure'])
            end
            print(fig2,[path,'figure\strain_stress.jpg'],'-djpeg','-r300');
        end        
    end
end

function [gamma,sxz,energy]=read_time_data(path,file)
    global ev2n
    fid=fopen([path,file]);
    tmp=cell2mat(textscan(fid,'%f %f %f %f','HeaderLines',1));
    sz=tmp(1:3)*1e-10;
    na=tmp(4);
    data=cell2mat(textscan(fid,'%f %f %f %f %f %f %f %f %f','HeaderLines',1));
    fclose(fid);
    tilt=data(:,2)*1e-10;
    gamma=tilt/sz(3);
    fx=data(:,3)*ev2n;
    sxz=-fx/(sz(1)*sz(2))/1e6;
    energy=data(:,9)/na;
end

function change=neigh_change(neigh0,neigh,n_neigh)
    change=[];
    ct=0;
    n=size(neigh,1);
    for i=1:n
        if sum(ismember(neigh(i,:),neigh0(i,:)))<n_neigh
            ct=ct+1;
            change(ct)=i;
        end
    end
end

function dpos=cal_dev_pos(tcrd,bcrd,neigh)
    dpos=[];
    n_neigh=size(neigh,2);
    for i=1:size(neigh,1)
        %disp(i)
        dpos(i)=sum((sum(bcrd(neigh(i,:),1:2),1)/n_neigh-tcrd(i,1:2)).^2);
        
        %disp(dpos(i));
        %plot(bcrd(neigh(i,:),1),bcrd(neigh(i,:),2),'.b')
        %hold on 
        %plot(tcrd(i,1),tcrd(i,2),'.r')   
        %hold off    
        
    end    
end

function [neigh,neigh2,bcrd]=find_neigh_pbc(tcrd,bcrd,cutoff,box,region,n_neigh)
    % % <neigh>: recorded in the extended number
    % % <neigh2>: recorded in the original id
    neigh=[];
    neigh2=[];
    xlo=box(1,1);xhi=box(1,2);ylo=box(2,1);yhi=box(2,2);
    % box correction due to insufficient accuracy
    xlo=min(xlo,min(min(tcrd(:,1)),min(bcrd(:,1))));
    ylo=min(ylo,min(min(tcrd(:,2)),min(bcrd(:,2))));
    xhi=max(xhi,max(max(tcrd(:,1)),max(bcrd(:,1))));
    yhi=max(yhi,max(max(tcrd(:,2)),max(bcrd(:,2))));   
    box=[xlo,xhi;ylo,yhi];
    
    xlen=xhi-xlo;
    ylen=yhi-ylo;
    
    natop=size(tcrd,1);
    nabot=size(bcrd,1);
    
    bcrd0=bcrd;
    % % map the image atoms in the bottom layer
    [bcrd,bmap]=image_atom(bcrd,cutoff,box);
    %plot(bcrd(:,1),bcrd(:,2),'ob')
    %hold on 
    %plot(bcrd0(:,1),bcrd0(:,2),'.r')
    
    % % get_subregion
    bsr=get_subregion(bcrd,region(1),region(2),cutoff,box);
    tsr=get_subregion(tcrd,region(1),region(2),0,box);
    
    for i=1:size(tsr,1)
        %plot(bcrd(bsr{i},1),bcrd(bsr{i},2),'.b')
        %hold on 
        %plot(tcrd(tsr{i},1),tcrd(tsr{i},2),'.r')   
        %hold off
        for j=1:numel(tsr{i,2})
            %disp([num2str(i),'  ',num2str(j)])
            delta=sum((tcrd(tsr{i,2}(j),1:2)-bcrd(bsr{i,1},1:2)).^2,2);
            [~,id]=sort(delta,'ascend');
            %neigh(tsr{i,2}(j),:)=bmap(id(1:3));
            neigh(tsr{i,2}(j),:)=bsr{i,1}(id(1:n_neigh));
            neigh2(tsr{i,2}(j),:)=bmap(bsr{i,1}(id(1:n_neigh)));
            %plot(tcrd(tsr{i,2}(j),1),tcrd(tsr{i,2}(j),2),'r.')
            %hold on
            %plot(bcrd(bsr{i,1}(id(1:n_neigh)),1),bcrd(bsr{i,1}(id(1:n_neigh)),2),'.b')
            %hold off            
        end
    end
end

function sr=get_subregion(crd,w,h,cutoff,box)
    xlo=box(1,1);xhi=box(1,2);ylo=box(2,1);yhi=box(2,2);
    xlen=xhi-xlo;
    ylen=yhi-ylo;
    sr={};
    n=0;
    xn=ceil(xlen/w);
    yn=ceil(ylen/h);
    for i=1:xn
        if i~=xn
            id=find( (crd(:,1)>=(i-1)*w-cutoff+xlo) & (crd(:,1)<i*w+cutoff+xlo) );
        else
            id=find( (crd(:,1)>=(i-1)*w-cutoff+xlo) & (crd(:,1)<=i*w+cutoff+xlo) );
        end
        crd1=crd(id,:);
        for j=1:yn
            if j~=yn
                id2=find( (crd1(:,2)>=(j-1)*h-cutoff+ylo) & (crd1(:,2)<j*h+cutoff+ylo) );
            else
                id2=find( (crd1(:,2)>=(j-1)*h-cutoff+ylo) & (crd1(:,2)<=j*h+cutoff+ylo) );
            end
            crd2=crd1(id2,:);
            if i~=xn 
                tmp1=and(crd2(:,1)>=(i-1)*w+xlo, crd2(:,1)<i*w+xlo);
            else
                tmp1=and(crd2(:,1)>=(i-1)*w+xlo, crd2(:,1)<=i*w+xlo);
            end            
            if j~=yn 
                tmp2=and(crd2(:,2)>=(j-1)*h+ylo, crd2(:,2)<j*h+ylo);
            else
                tmp2=and(crd2(:,2)>=(j-1)*h+ylo, crd2(:,2)<=j*h+ylo);
            end
            id3=find(tmp1 & tmp2);
            n=n+1;
            sr{n,1}=id(id2);
            sr{n,2}=id(id2(id3));
        end
    end
end

function [crd,map]=image_atom(crd,cutoff,box)
    xlo=box(1,1);xhi=box(1,2);ylo=box(2,1);yhi=box(2,2);
    xlen=xhi-xlo;
    ylen=yhi-ylo;
    nabot=size(crd,1);
    nabot0=nabot;
    map=[1:nabot]';
    
    id1=find(abs(crd(1:nabot0,1)-xlo)<=cutoff);
    n=numel(id1);
    crd(nabot+1:nabot+n,:)=crd(id1,:);
    crd(nabot+1:nabot+n,1)=crd(nabot+1:nabot+n,1)+xlen;
    map(nabot+1:nabot+n)=id1;
    nabot=nabot+n;
    
    id2=find(abs(crd(1:nabot0,1)-xhi)<=cutoff);
    n=numel(id2);
    crd(nabot+1:nabot+n,:)=crd(id2,:);
    crd(nabot+1:nabot+n,1)=crd(nabot+1:nabot+n,1)-xlen;
    map(nabot+1:nabot+n)=id2;
    nabot=nabot+n;   
    
    id3=find(abs(crd(1:nabot0,2)-ylo)<=cutoff);   
    n=numel(id3);
    crd(nabot+1:nabot+n,:)=crd(id3,:);
    crd(nabot+1:nabot+n,2)=crd(nabot+1:nabot+n,2)+ylen;
    map(nabot+1:nabot+n)=id3;
    nabot=nabot+n;
    
    id4=find(abs(crd(1:nabot0,2)-yhi)<=cutoff);
    n=numel(id4);
    crd(nabot+1:nabot+n,:)=crd(id4,:);
    crd(nabot+1:nabot+n,2)=crd(nabot+1:nabot+n,2)-ylen;
    map(nabot+1:nabot+n)=id4;
    nabot=nabot+n;    
    
    id5=intersect(id1,id3);
    n=numel(id5);
    crd(nabot+1:nabot+n,:)=crd(id5,:);
    crd(nabot+1:nabot+n,1)=crd(nabot+1:nabot+n,1)+xlen;
    crd(nabot+1:nabot+n,2)=crd(nabot+1:nabot+n,2)+ylen;
    map(nabot+1:nabot+n)=id5;
    nabot=nabot+n;      
    
    id6=intersect(id1,id4);
    n=numel(id6);
    crd(nabot+1:nabot+n,:)=crd(id6,:);
    crd(nabot+1:nabot+n,1)=crd(nabot+1:nabot+n,1)+xlen;
    crd(nabot+1:nabot+n,2)=crd(nabot+1:nabot+n,2)-ylen;
    map(nabot+1:nabot+n)=id6;
    nabot=nabot+n; 
    
    id7=intersect(id2,id3);
    n=numel(id7);
    crd(nabot+1:nabot+n,:)=crd(id7,:);
    crd(nabot+1:nabot+n,1)=crd(nabot+1:nabot+n,1)-xlen;
    crd(nabot+1:nabot+n,2)=crd(nabot+1:nabot+n,2)+ylen;
    map(nabot+1:nabot+n)=id7;
    nabot=nabot+n;     
    
    id8=intersect(id2,id4);
    n=numel(id8);
    crd(nabot+1:nabot+n,:)=crd(id8,:);
    crd(nabot+1:nabot+n,1)=crd(nabot+1:nabot+n,1)-xlen;
    crd(nabot+1:nabot+n,2)=crd(nabot+1:nabot+n,2)-ylen;
    map(nabot+1:nabot+n)=id8;
    nabot=nabot+n;
    
end

function [topid,botid]=detect_dislocation_interface(coord)
    topid=[];
    botid=[];
    dev=0.1;
    coord_sort=sort(coord(:,3),'descend');
    id=find(coord_sort>=(0-dev));
    h=coord_sort(id(end));
    topid=find(coord(:,3)<=h+dev & coord(:,3)>=(h-dev));
    id=find(coord_sort<h-dev);
    h=coord_sort(id(1));
    botid=find(coord(:,3)<=h & coord(:,3)>=(h-dev));       
end

function curnt=remap_cross_zpbc(ref,curnt,lz)
    thre=lz*0.5;
    delta=curnt(:,3)-ref(:,3);
    id=find(delta>thre);
    curnt(id,3)=curnt(id,3)-lz;
    id2=find(delta<-thre);
    curnt(id2,3)=curnt(id2,3)+lz;    
end

function [step,natom,id,coord,box,mapid]=read_data(fid,read_type,target,ifrecursive)
    count=0;
    ct=0;
    step=[];
    natom=[];
    id={};
    coord={};
    box={};
    mapid={};
    max_target=max(target);
    ifread=zeros(1,numel(target));
    while ~feof(fid)
        count=count+1;
        
        if strcmp(read_type,'frame')
            [ismem,loc]=ismember(count,target);
            if numel(loc)>1
                disp('warning: repeated target')
            end
            if ismem
                [step_tmp,natom_tmp,id_tmp,coord_tmp,box_tmp]=read_frame(fid,0);
                ct=ct+1;
                step(ct)=step_tmp;
                natom(ct)=natom_tmp;
                id{ct}=id_tmp;
                coord{ct}=coord_tmp;
                box{ct}=box_tmp;
                mapid{ct}=map_id(id{ct});
                ifread(loc)=1;
            else
                [~,~,~,~,~]=read_frame(fid,0);
            end
            if ~ifrecursive && count>max_target
                break;
            end
        elseif strcmp(read_type,'step')
            [step_tmp,natom_tmp,id_tmp,coord_tmp,box_tmp]=read_frame(fid,0);
            [ismem,loc]=ismember(step_tmp,target);
            if numel(loc)>1
                disp('warning: repeated target')
            end
            if ismem
                ct=ct+1;
                step(ct)=step_tmp;
                natom(ct)=natom_tmp;
                id{ct}=id_tmp;
                coord{ct}=coord_tmp;
                box{ct}=box_tmp;
                mapid{ct}=map_id(id{ct});
            end    
            if ~ifrecursive && step_tmp>max_target
                break;
            end
        end
    end   
 
end   
    
function [step,natom,id,coord,box]=read_frame(fid,ifskip)
    box=[];
    coord=[];
    id=[];
    % read the initial frame to check if initial fail occur and if the 1st bond break
    tmp=textscan(fid,'%d',1,'HeaderLines',1);
    step=tmp{1,1}(1);
    tmp=textscan(fid,'%d',1,'HeaderLines',2);
    natom=tmp{1,1}(1);     % number of atom
    tmp=textscan(fid,'%f %f',3,'HeaderLines',2);
    xlo=tmp{1,1}(1);
    xhi=tmp{1,2}(1);
    ylo=tmp{1,1}(2);
    yhi=tmp{1,2}(2);
    zlo=tmp{1,1}(3);        % box low boundary
    zhi=tmp{1,2}(3);        % box high boundary
    box=[xlo,xhi;ylo,yhi;zlo,zhi];    

    if ifskip
        tmp=textscan(fid,'%s',1,'HeaderLines',2+natom);
    elseif ~ifskip
        tmp=textscan(fid,'%d %d %f %f %f','HeaderLines',2);
        id=tmp{1,1};            % atom id, no order
        coord(:,1)=tmp{1,3};     % x coordinate
        coord(:,2)=tmp{1,4};     % y coordinate
        coord(:,3)=tmp{1,5};     % z coordinate
        
        if size(coord,1)~=natom
            disp(['n_atom not equal lines of data at file ',num2str(i),' initial frame'])
            bond=[];
            return;
        end
        [id,seq]=sort(id);
        coord=coord(seq,:);
    end
end

function map=map_id(id)
    map=[];
    for i=1:size(id,1)
        map(id(i))=i;
    end
end

function id=get_dataset_id(ds,name)
    id=0;
    for i=1:numel(ds)
        if strcmp(name,ds(i).name)
            id=i;
            break;
        end
    end
end