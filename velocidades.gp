set terminal png size 400,300 enhanced font "Helvetica,20"
set output 'output.png'
n=100 #number of intervals
max=3. #max value
min=-3. #min value
width=(max-min)/n #interval width
#function used to map a value to the intervals
hist(x,width)=width*floor(x/width)+width/2.0
set boxwidth width*0.9
set style fill solid 0.5 # fill style

#count and plot
plot "datos2.txt" u (hist($1,width)):(1.0) smooth freq w boxes lc rgb"green" notitle,
plot "datos2.txt" u (hist($3,width)):(1.0) smooth freq w boxes lc rgb"red" notitle
