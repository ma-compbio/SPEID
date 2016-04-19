# Installing theano, keras, and their dependencies was surprisingly painful, so
# here's a script that should accomplish this for you
# Note that this shell script was written after I did this directly on the
# command line, so I haven't tested it, and am not sure that it works.

# Download and install Python
wget https://www.python.org/ftp/python/2.7.11/Python-2.7.11.tar.xz
tar -xvf Python-2.7.11.tar.xz
rm Python-2.7.11.tar.xz
cd Python-2.7.11/
./configure --enable-shared --prefix=$HOME/local
make
make install
cd ..

# point python to its shared libraries
export LD_LIBRARY_PATH=$HOME/local/lib

# make an alias to run the new python installation
alias python2.7=$HOME/local/bin/python2.7

# now is a good point to run 'python2.7 -V' to check that the install worked

# Install HDF5 (necessary for reading/writing models)
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.gz
tar -xf hdf5-1.8.16.tar.gz
cd hdf5-1.8.16
# We probably can remove --disable-parallel --without-szlib --without-pthread
./configure --disable-parallel --without-szlib --without-pthread --prefix=$HOME/local
make
make install
cd ..
# Point python to HDF5 installation
export HDF5_DIR=$HOME/local

# fortunately, newer versions of Python come packaged with pip
# this step takes a while (especially scipy)
python2.7 -m pip install numpy scipy nose theano h5py keras==0.2.0

git clone https://github.com/xianyi/OpenBLAS  
cd OpenBLAS  
make FC=gfortran
make PREFIX=$HOME/local install
cd ..

# create a file called .theanorc with the following contents
# [blas]
# ldflags = -L$HOME/local/lib -lopenblas

# The numpy test will probably fail after about 500 test.
# This is OK (there is a bug in the test, regarding where f2py should be, due
# to the details of out local setup). Ignore the many, many warnings...
python2.7 -c "import numpy; numpy.test()" # Takes ~1m and fails
python2.7 -c "import scipy; scipy.test()" # Takes ~5m
python2.7 -c "import theano; theano.test()" # Takes ~1h

# In order for the Python to work next time you log in, add the following to
# your .bashrc:
#
# PYTHONPATH="${PYTHONPATH}:{$HOME}/local/lib/python2.7/site-packages/"
# export PYTHONPATH
# export LD_LIBRARY_PATH=$HOME/local/lib
# alias python2.7=$HOME/local/bin/python2.7
