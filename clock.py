import numpy as np

h = range(12)
print('hh:mm:ss ss_fraction')
for c in h:
	for d in h:
		num_t = 516240*c+43920*d
		den_t = 143
		r = num_t%den_t
		q = num_t/den_t
		hrs = q/3600
		q = q - 3600*hrs
		mns = q/60
		q = q - 60*mns
		if r > 0:
			print(('%02d:%02d:%02d'%(hrs,mns,q))+' '+str(r)+'/143')
		else:
			print('%02d:%02d:%02d'%(hrs,mns,q))

