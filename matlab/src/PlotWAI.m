function PlotWAI(Tb,ftype,Title)
    [ftype,isComplex,c] = setWavAndColors(ftype);
    no_plot = 2; 
    lgd = {'positive','negative'};
    if isComplex, no_plot = no_plot +1; end
    figure
    % Plot signal and wavelet
    ax1 = subplot(no_plot,1,1);
    plot(Tb.time,Tb{:,1},'Color',[0.9 0.9 0.9],'LineStyle','-')
    ylabel('Signal')
    yyaxis right
    if isComplex
        plot(Tb.time,Tb.(strcat(ftype,"_re")),'Color',c(1,:),'LineStyle','-','LineWidth',2)
        hold on
        plot(Tb.time,Tb.(strcat(ftype,"_im")),'Color',c(4,:),'LineStyle','--','LineWidth',2)
    else
        plot(Tb.time,Tb.(ftype),'Color',c(1,:),'LineStyle','-','LineWidth',2)
    end
    ylabel('Wavelet')
    set(ax1,'YColor',[0.15 0.15 0.15])
    if ~isempty(Title)
        title(Title)
    end
    
    % Plot Area (real part)
    ax2 = subplot(no_plot,1,2);
    area(Tb.time,Tb.(strcat("PosRes",ftype,"_re")),'FaceColor',c(2,:),'EdgeColor',c(2,:))
    hold on 
    area(Tb.time,Tb.(strcat("NegRes",ftype,"_re")),'FaceColor',c(3,:),'EdgeColor',c(3,:))
    ylabel('Product curve')
    legend(lgd(sum(abs(Tb{:,strcat(["Pos","Neg"],"Res",ftype,"_re")}))>1e-5))
    linkaxes([ax1,ax2],'x')

    % Plot Area (imaginary part)
    if isComplex
        ax3 = subplot(no_plot,1,3);
        area(Tb.time,Tb.(strcat("PosRes",ftype,"_im")),'FaceColor',c(5,:),'EdgeColor',c(5,:))
        hold on 
        area(Tb.time,Tb.(strcat("NegRes",ftype,"_im")),'FaceColor',c(6,:),'EdgeColor',c(6,:))
        ylabel('Product curve (imaginary part)')
        legend(lgd(sum(abs(Tb{:,strcat(["Pos","Neg"],"Res",ftype,"_im")}))>1e-5))
        linkaxes([ax1,ax2,ax3],'x')
    end
    
end



function [ftype,isComplex,c] = setWavAndColors(ftype)
% Format and nature (complex/real) of ftype & definition of colors

% Wavelet (ftype) format
ftype = char(upper(ftype));
if ftype<3
    error('Wavelet "%s" inadmissible. \nPlease select an option among Morlet, Mexhat, DOGx (x=order of derivation), Gauss or Haar',ftype)
end
if ~ismember(ftype(1:3),{'MOR','MEX','HAA','DOG','GAU'}) % Haar
    error('Wavelet "%s" inadmissible. \nPlease select an option among Morlet, Mexhat, DOGx (x=order of derivation), Gauss or Haar',ftype)
end
if strcmp(ftype(1:3),'DOG') % For DOG wavelets
    if isempty(regexp(ftype, '^DOG\d+$', 'once'))
        error('Derivation order of "%s" inadmissible. \nPlease sepcify a numeric derivation order',ftype)
    end
else
    ftype = ftype(1:3);
end
% Conversion of Mexican Hat to DOG2
if strcmp(ftype(1:3),'MEX'), ftype = 'DOG2';end
% Conversion of DOG0 to Gauss
if sscanf(ftype,'DOG%d')==0, ftype = 'GAU';end

% Define colors and if complex wavelet
isComplex = false;
switch ftype(1:3)
    case 'MOR' % Morlet wavelet
        isComplex = true;
        cwav_re = [0,0.45,0.74];
        cwav_im = [0.85 0.33 0.1];
        cpos_re = [0.07 0.62 1];
        cpos_im = [1,0.41,0.16];
        cneg_re = [0.05,0.24,0.36];
        cneg_im = [0.44,0.05,0.12];
        c = [cwav_re;cpos_re;cneg_re;cwav_im;cpos_im;cneg_im];
    case 'GAU' % Gauss function
        cwav = [0.64,0.08,0.18];
        cpos = [0.64,0.08,0.18];
        cneg = [0.64,0.08,0.18]; % No negative color
        c = [cwav;cpos;cneg];
    case 'DOG' % Gaussian derivatives 
        m = sscanf(ftype,'DOG%d'); % Order of the derivative
        if m==1
            cwav = [0.16,0.16,0.61];
            cpos = [0.07,0.64,0.95];
            cneg = [0.05,0.05,0.47];
            c = [cwav;cpos;cneg];
        else
            cwav = [0.47,0.67,0.19];
            cpos = [0.39,0.83,0.07];
            cneg = [0.47,0.67,0.19];
            c = [cwav;cpos;cneg];
        end
    case 'HAA' % Haar wavelet
        cwav = [0.49,0.18,0.56];
        cpos = [0.72,0.27,1.00];
        cneg = [0.28,0.12,0.32];
        c = [cwav;cpos;cneg];
end
end