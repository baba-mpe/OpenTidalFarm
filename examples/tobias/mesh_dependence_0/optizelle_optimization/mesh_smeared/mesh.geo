basin_x = 2000;
basin_y = 1000;
site_x = 500;
site_y = 300;
element_size = 20;
element_size_coarse = 50;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};
Point(3) = {0, basin_y, 0, element_size_coarse};
Point(4) = {basin_x, basin_y, 0, element_size_coarse};

Point(5) = {(basin_x - site_x)/2, (basin_y - site_y)/2, 0, element_size};
Point(6) = {(basin_x - site_x)/2+site_x, (basin_y - site_y)/2, 0, element_size};
Point(7) = {(basin_x - site_x)/2, (basin_y - site_y)/2+site_y, 0, element_size};
Point(8) = {(basin_x - site_x)/2+site_x, (basin_y - site_y)/2+site_y, 0, element_size};

Line(6) = {1, 2};
Line(7) = {2, 4};
Line(8) = {4, 3};
Line(9) = {3, 1};
Line Loop(10) = {9, 6, 7, 8};
Line Loop(11) = {3, 2, -4, -1};

Physical Line(2) = {7};
Physical Line(1) = {9};
Physical Line(3) = {8, 6};
Line(12) = {7, 8};
Line(13) = {6, 8};
Line(14) = {5, 6};
Line(15) = {7, 5};
Line Loop(16) = {15, 14, 13, -12};
Plane Surface(17) = {10, 16};
Plane Surface(18) = {16};
Physical Surface(2) = {17};
Physical Surface(1) = {18};
