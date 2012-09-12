import os
import shutil
import argparse
import string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    
    # required arguments
    aa('--directory',
       help='the directory where your new app will live',
       required = True)
    aa('--source',
       help='the name of your application',
       required = True)
    
    # optional arguments
    aa('--filesystem',
       action = 'store_true',
       help = 'use the FileSystemFrameController')
    aa('--pytables',
       action = 'store_true',
       help = 'use the PyTablesFrameController')
    
    args = parser.parse_args()
    if os.path.exists(args.directory):
        # Make sure the user really wants to delete the existing app
        s =raw_input('A directory already exists at % s. \
            Are you sure you\'d like to delete its contents?' % args.directory)
        if 'y' != s:
            # They don't. exit.
            exit()
        # They do. Remove the directory and its contents.
        shutil.rmtree(args.directory)
    
    datadir = 'datastore'
    # Create directories for the app, datastore, and logs.
    os.makedirs(args.directory)
    os.makedirs(os.path.join(args.directory,datadir))
    os.makedirs(os.path.join(args.directory,'log'))
    
    dr = args.directory
    # get the zounds installation path.
    import zounds
    zp = os.path.join(zounds.__path__[0],'quickstart')
    # copy the files to the new app directory
    for fn in os.listdir(zp):
        path = os.path.join(zp,fn)
        if fn != os.path.split(__file__)[1] and os.path.isfile(path):
            shutil.copy(path,os.path.join(dr,fn))
    
    # copy and setup the websearch stuff
    # KLUDGE: Templates shouldn't go into the static directory
    ws = 'websearch'
    st = 'static'
    src_wsdir = os.path.join(zp,ws)
    dst_wsdir = os.path.join(args.directory,st)
    # copy the entire directory
    shutil.copytree(src_wsdir,dst_wsdir)
    # move websearch.py into the root folder of the app
    wspy = '%s.py' % ws
    old_fn = os.path.join(dst_wsdir,wspy)
    new_fn = os.path.join(args.directory,wspy)
    shutil.move(old_fn,new_fn)
    
    # Read the contents of the config file template
    configfile = os.path.join(dr,'config.py')
    with open(configfile,'r') as f:
        config_t = string.Template(f.read())
    
    # get appropriate values based on the user's choice of FrameController
    controller_class_name = \
        'FileSystemFrameController' if args.filesystem else 'PyTablesFrameController'
    db_filename = datadir if args.filesystem else os.path.join(datadir,'frames.h5')
    # Substitute the user-specified parameters in the config file
    s = config_t.substitute({'Source'              : args.source,
                             'ControllerClassName' : controller_class_name,
                             'DbFile'              : db_filename})
    
    # Write the real config file to disk
    with open(configfile,'w') as f:
        f.write(s)
    
    
        
    
    
    