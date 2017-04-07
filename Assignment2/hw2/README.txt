In order to complete this homework you will need Python 3 and Theano installed.
One very easy way to get this environment installed is to use Anaconda:
https://www.continuum.io/downloads
Please note that the Jupyter notebook is written in Python 3. 

For a quickstart with Jupyter:
https://jupyter.readthedocs.io/en/latest/content-quickstart.html

To start working on the homework, in the directory in which you
unpacked the homework files run:
	jupyter notebook hw2.ipynb

This will open a browser window that you can use as a development environment.

'sha1:331bd0d4cfca:deefaa15057b1d702bb78fc8614fb9a8a8f207e4'



c = get_config()

# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook

# Notebook config
c.NotebookApp.certfile = u'/home/ec2-user/certs/mycert.pem' #location of your certificate file
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False  #so that the ipython notebook does not opens up a browser by default
c.NotebookApp.password = u'sha1:331bd0d4cfca:deefaa15057b1d702bb78fc8614fb9a8a8f207e4'  #the encrypted password we generated above
# It is a good idea to put it on a known, fixed port
c.NotebookApp.port = 8888
  

conda install theano pygpu

https://chrisalbon.com/jupyter/run_project_jupyter_on_amazon_ec2.html
http://deeplearning.net/software/theano/install_ubuntu.html