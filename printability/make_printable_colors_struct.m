% https://github.com/mahmoods01/accessorize-to-a-crime/tree/master/aux/attack
function printable_colors = make_printable_colors_struct(printed_im_path, num_colors)
% Make a Nx3 matrix of all the RGB values that exist in a printed
% image (preferably a printed palette that covers a large portion
% of the RGB space).

    printed_im = imread(printed_im_path);
    
    % cut 3% of the edges from each side (in case of misalignment of the
    % printed image)
    cut_h = round(0.015*size(printed_im,2));
    cut_v = round(0.015*size(printed_im,1));
    printed_im = printed_im(1+cut_v-1:end-cut_v, 1+cut_h-1:end-cut_h, :);

    % the printable_colors matrix is actually the color map found with the
    % minimum variance approach
    if nargin==1
        [~,printable_colors] = rgb2ind(printed_im, 1024);
    else
        [~,printable_colors] = rgb2ind(printed_im, num_colors);
    end
    
    % {0,...,255} range and sort by column
    printable_colors = round(printable_colors*255);
    printable_colors = sortrows(printable_colors);

end