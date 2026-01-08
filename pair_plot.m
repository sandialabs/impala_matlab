function pair_plot(meas, label, truth)
    % pairplot(mat, label)
    %
    % pairwise scatter & histogram different colors with group
    %
    arguments
        meas
        label
        truth=[]
    end
    label1 = label;
    label1{length(label)+1} = 'Color';
    tbl = array2table([meas (1:size(meas,1))'], 'VariableNames', label1);
    for i = 1:size(meas, 2)
        for j = 1:size(meas, 2)
            subAx = subplot(size(meas, 2),size(meas, 2),sub2ind([size(meas, 2) size(meas, 2)], i, j));
            if i==1
                ylabel(label{j});
            end
            if j==size(meas, 2)
                xlabel(label{i});
            end
            hold on;
            if i==j
                bin = linspace(min(meas(:,i)), max(meas(:,i)), 20);
                histogram(meas(:, i), bin, 'Normalization', 'pdf', ...
                    'FaceColor', [152/255, 185/255, 214/255], 'FaceAlpha', 1);
                hold on
                [f,x] = ksdensity(meas(:, i),'function','pdf');
                plot(x,f, 'Linewidth', 2, 'Color', [60/255, 117/255, 175/255])
                xline(median(meas(:,i)), '--k', 'LineWidth', 2)
                if ~isempty(truth)
                    xline(truth(i), '--r', 'LineWidth', 2)
                end
                xlim([bin(1) bin(end)])
            elseif i>j
                scatter(tbl,label{i}, label{j}, 'filled', 'ColorVariable', 'Color', 'SizeData', 18)
                colormap(subAx,coolwarm)
                xlim([min(meas(:, i)) max(meas(:, i))])
            else
                [f,x] = ksdensity(meas(:,[i,j]),'function','pdf');
                x1 = x(:,1);
                x2 = x(:,2);
                x = linspace(min(x1),max(x1));
                y = linspace(min(x2),max(x2));
                [xq,yq] = meshgrid(x,y);
                z = griddata(x1,x2,f,xq,yq);
                thresh = mean(z(:)) - .05*std(z(:));
                z(z<=thresh) = NaN;
                contourf(xq,yq,z)
                hold on
                scatter(median(meas(:,i)), median(meas(:,j)), 'Marker', 'x', 'SizeData', 100, 'LineWidth', 2, 'MarkerEdgeColor', 'k')
                if ~isempty(truth)
                    scatter(truth(i), truth(j), 'Marker', 'x', 'SizeData', 100, 'LineWidth', 2, 'MarkerEdgeColor', 'r')
                end
            end
        end
    end
    
    