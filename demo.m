function demo()

a = load('2018-02-03-06-34-30-detected-persons.txt.mat');
b = load('2018-02-03-06-34-30-laser-scan.txt.mat');

ref_time = a.data(1,1);


a.data(a.data(:,2) == -999, :) = [];
ptime = (a.data(:,1) - ref_time) ; 
py = a.data(:,2);
px = a.data(:,3);

ltime = (b.data(:,1) - ref_time);
lrange = b.data(:,2:end);

%  The min and max angles are 
% -1.69296944141 and 1.6929693222 radians 
% corresponding to the first and last array values.  
% The angle increment is 0.00872664619237 radians.  

angle = -1.69296944141:0.00872664619237:1.6929693222;

% in_view = angle > -0.5 & angle < 0.5;

in_view = angle > -9.5 & angle < 9.5;


ly = lrange.*cos(angle); 
lx = lrange.*sin(angle); 


i_interp = interp1(ltime, 1:length(ltime), ptime, 'nearest');
lti = ltime(i_interp);
lxi = lx(i_interp, :);
lyi = ly(i_interp, :);


figure; hold all; 
prev = 0;
count = 0;
axis([min(lx(:)) max(lx(:)) min(ly(:)) max(ly(:))]);
set(gca, 'xdir', 'reverse')
% ze1 = zeros(size(lxi(1,~in_view)));
% na1 = nan(size(lxi(1,~in_view)));
% ze2 = zeros(size(lxi(1,in_view)));
% na2 = nan(size(lxi(1,in_view)));
% ph1a = plot([ze1 lxi(1,~in_view) ], lyi(1,~in_view), '.r');
% ph1b = plot(lxi(1,in_view), lyi(1,in_view), '.b');
ph1a = rayplot(lxi(1,~in_view), lyi(1,~in_view), '-xr');
ph1b = rayplot(lxi(1,in_view), lyi(1,in_view), '-xb');

cam_y = -0.5312;
my = max(ly(:));
plot([0 my*tan(0.5) NaN 0 -my*tan(0.5)], ...
     [cam_y my NaN cam_y my], '-r')


ph2 = []; 
for i = 1:1:length(lti)
    if ptime(i) ~= prev
        pause(0.01)
%         input('return to continue')
%         clf; hold all;
        for j = 1:length(ph2)
            delete(ph2(j))
        end
        ph2 = []; 

        set(ph1a, 'Xdata', rays(lxi(i,~in_view)))
        set(ph1a, 'Ydata', rays(lyi(i,~in_view)))
        
        set(ph1b, 'Xdata', rays(lxi(i,in_view)))
        set(ph1b, 'Ydata', rays(lyi(i,in_view)))

        title(sprintf('%.2f, %.2f, %d', ptime(i), ...
                       lti(i), count))
        prev = ptime(i); 
        count = 0;
    end
    
    count = count + 1;
    ph2(end+1) = plot(px(i), py(i), 'o'); %#ok<AGROW>
    disp(length(lti) - i)
%     input('return to continue')
end

function p = rayplot(x, y, varargin)

x = rays(x);
y = rays(y);

p = plot(x, y, varargin{:});

function x = rays(x)

x = x(:);
ze = zeros(size(x));
na = nan(size(x));

x = [ze x na]';
x = x(:);

