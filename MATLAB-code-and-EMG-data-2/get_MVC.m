function Norm_data = get_MVC(data, Peak)
    %Power spectrum
    [P,f] = pspectrum(data,200);
    plot(f,pow2db(P));
    
    % Detrend data
    Detrend_Data = detrend(data);
    
    
    % Rectify filtered data
    Rec_filtered_data = abs(Detrend_Data);
    
    % Apply moveing average
    Movav_data = movmean(Rec_filtered_data,150,1);
    
    % Find peak MVC
    Peak = max(Mvc_data);
    
    % Normalize data
    Norm_data = Movav_data/Peak*100; 
end



