set terminal png size 400,300 enhanced font "Helvetica,20"
set output 'output_fase.png'
n=100 #number of intervals
max=3. #max value
min=-3. #min value
width=(max-min)/n #interval width
#function used to map a value to the intervals

#count and plot
 plot "datos2.txt" u 1:2 with dots lc rgb "red"

