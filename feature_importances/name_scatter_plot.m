function scatter_names(x, y, names)

  scatter(x, y);

  % displacement so text does not overlay the data points
  dx = (max(x) - min(x))/100;
  dy = (max(y) - min(y))/100;

  text(x+dx, y+dy, names);

end
