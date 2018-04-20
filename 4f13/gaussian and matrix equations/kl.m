clf
hold off
[y h1] = draw_ellipse([0 0], [1 0.9; 0.9 1], 'g');
hold on
[y h2] = draw_ellipse([0 0], [1 0; 0 1], 'b', '-.');
[y h3] = draw_ellipse([0 0], [0.18 0; 0 0.18], 'r', '--');
set(gca,'FontSize',18);
axis equal
legend([h1(1) h2(1) h3(1)],'\Sigma_g','\Sigma_1','\Sigma_2','Location','SouthEast')
