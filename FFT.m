%% Import data
clear all
load('Ins1Whole_ML040218.mat', 'NewDataTruncated')
Data = NewDataTruncated;
clear NewDataTruncated
fid = fopen('ins1.txt');
text = textscan(fid,'%s%s%s%s%s');
fclose(fid);

%% get FFT
low = [0.5,4,8.5,12.5,16];
high = [3.5,8,12,16,30];
bracket = {'FFT','delta', 'theta', 'alpha', 'sigma', 'beta'};
bracket{2,1} = ''; bracket{3,1} = 'Minimum'; bracket{4,1} = 'Maximum';bracket{5,1} = 'Average';

Fs = 256;
L = Fs*30; %30 Seconds
Frequencies = [];
% disp(size(Data(1,2:L+1)));
for k = [1:5]
    for i = 1:L:length(Data)-L
        Y = fft(Data(k,1+i:L+i));
        P2 = abs(Y);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        f = Fs*(0:(L/2))/L;
    %     plot(f,P1) 
    %     title('Single-Sided Amplitude Spectrum of X(t)')
    %     xlabel('f (Hz)')
    %     ylabel('|P1(f)|')

        % Frequency brackets
        for j=2:6
            bracket{2,j} = P1(find(~(f-low(j-1))):find(~(f-high(j-1))));
            if i == 1
                bracket{3,j} = min(bracket{2,j}); % find min value of the bracket
                bracket{4,j} = max(bracket{2,j}); % find min value of the bracket
                bracket{5,j} = mean(bracket{2,j}); % find min value of the bracket
            else
                bracket{3,j} = [bracket{3,j}; min(bracket{2,j})]; % find min value of the bracket
                bracket{4,j} = [bracket{4,j}; max(bracket{2,j})]; % find min value of the bracket
                bracket{5,j} = [bracket{5,j}; mean(bracket{2,j})]; % find min value of the bracket
            end
        end
    
    end
    for ii = 2:6
        Frequencies = [Frequencies,bracket{5,ii}];
    end
end
alpha = Frequencies(:,3:5:25);
beta = Frequencies(:,5:5:25);

% fid = fopen('data.csv','wt');
% if fid>0
%     for row = 3:5
%         for col = 2:6
%             fprintf(fid,'%s,%f\n',bracket{row,col});
%         end
%         fclose(fid);
%     end
% end

% csvwrite('deltamin.csv',bracket{3,2}');
% csvwrite('deltamax.csv',bracket{4,2}');
% csvwrite('deltamean.csv',bracket{5,2}');
% csvwrite('thetamin.csv',bracket{3,3}');
% csvwrite('thetamax.csv',bracket{4,3}');
% csvwrite('thetamean.csv',bracket{5,3}');
% csvwrite('alphamin.csv',bracket{3,4}');
% csvwrite('alphamax.csv',bracket{4,4}');
% csvwrite('alphamean.csv',bracket{5,4}');
% csvwrite('sigmamin.csv',bracket{3,5}');
% csvwrite('sigmamax.csv',bracket{4,5}');
% csvwrite('sigmamean.csv',bracket{5,5}');
% csvwrite('betamin.csv',bracket{3,6}');
% csvwrite('betamax.csv',bracket{4,6}');
% csvwrite('betamean.csv',bracket{5,6}');
% scatter(Frequencies(:,3),Frequencies(:,5))
% title('PCA space representation of EEG data')
% xlabel('Alpha')   
% ylabel('Beta')

% %% PCA
% [coeff,score,latent] = pca(Frequencies);
% % [coeff,score,latent] = pca([alpha,beta]);
% pcaspace = Frequencies*coeff;
% % pcaspace = [alpha,beta]*coeff;
% scatter3(pcaspace(:,1),pcaspace(:,2),pcaspace(:,3))
% title('PCA space representation of EEG data')
% xlabel('PCA1')
% ylabel('PCA2')