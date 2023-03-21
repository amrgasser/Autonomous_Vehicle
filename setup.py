from distutils.core import setup

setup(name='Distutils',
      version='1.0',
      description='Self driving car on MIT VISTA simulator',
      author='Amr ElSayed',
      author_email='amr_gasser@hotmail.com',
      packages=[
          'tensorflow',
          'numpy',
          'matplotlib',
          'cv2',
          'gym',
          'tqdm',
          'tensorflow_probability==0.12.0',
          'scikit-video',
          'ffio',
          'shapely==1.8.5',
          'pyrender',
          'descartes',
          'pyvirtualdisplay',
          'git+https://github.com/vista-simulator/vista-6s191.git',
          'descartes',
          'vista',
          'tensorflow_probability==0.12.0'
      ],
      )
