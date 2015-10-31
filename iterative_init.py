import matplotlib 
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, '/home/dmishkin/dev/irish-coffe/python')
if len (sys.argv) < 4:
  raise RuntimeError('Usage: python ' + sys.argv[0] + ' path_to_solver path_to_save_model mode')
import caffe

from pylab import *
import random
import numpy as np

gpu_runline = list()



solver_path = str(sys.argv[1])
init_path = str(sys.argv[2])
init_mode =  str(sys.argv[3])
margin = 0.01;
max_iter = 100;


if len (sys.argv) >= 5:
  neg_slope =  float(str(sys.argv[4]))
else:
  neg_slope=0;

mode_check=False;  
if init_mode == 'Orthonormal':
  mode_check=True
elif init_mode == 'IterNorm':
  mode_check=True
elif init_mode == 'OrthonormalIterNorm':
  mode_check=True
elif init_mode == 'OrthonormalIterNormMean':
  mode_check=True
else:
  raise RuntimeError('Unknown mode. Try Orthonormal or IterNorm or  OrthonormalIterNorm')

caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_path)
if os.path.isfile(init_path):
  print "Loading"
  solver.net.copy_from(init_path)

def svd_orthonormal(shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q

for k,v in solver.net.params.iteritems():
    print k, v[0].data.shape,v[1].data.shape
    if 'Orthonormal' in init_mode:
      weights=svd_orthonormal(v[0].data[:].shape)
      solver.net.params[k][0].data[:]=weights#* sqrt(2.0/(1.0+neg_slope*neg_slope));
      if 'Mean' in init_mode:
	biases=solver.net.params[k][1].data[:]
	biases[:]=0;
	solver.net.params[k][1].data[:]=biases;
    else:
      weights=solver.net.params[k][0].data[:]
      if 'Mean' in init_mode:
	biases=solver.net.params[k][1].data[:]
    if 'IterNorm' in init_mode:
      solver.net.forward()
      v = solver.net.blobs[k];
      var1=np.var(v.data[:]);
      mean1 = np.mean(v.data[:]);
      print k,'var = ', var1,'mean = ', mean1
      sys.stdout.flush()
      iter_num = 0;
      while (abs(1.0 - var1) > margin):
	if 'Mean' in init_mode:
	  biases=solver.net.params[k][1].data[:]
	  solver.net.params[k][1].data[:] -= mean1 / len(biases[:]); 
	weights=solver.net.params[k][0].data[:]
	solver.net.params[k][0].data[:] = weights / sqrt(var1);
        solver.net.forward()
        v = solver.net.blobs[k];
        var1=np.var(v.data[:]);
        mean1 = np.mean(v.data[:]);
        print k,'var = ', var1,'mean = ', mean1
        sys.stdout.flush()
        iter_num+=1;
        if iter_num > max_iter:
	  print 'Could not converge in ', iter_num, ' iterations, go to next layer'
	  break;
        
print "Initialization finished!"
for k,v in solver.net.params.iteritems():
    vv = solver.net.blobs[k];
    print k,vv.data[:].shape, ' var = ', np.var(vv.data[:]), ' mean = ', np.mean(vv.data[:]),
print "Saving model..."
solver.net.save(init_path)
print "Model saved to", init_path
