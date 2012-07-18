import os
import shutil
import argparse
import string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    
    aa('--directory',help='the directory where your new app will live')
    aa('--source',help='the name of your application')
    
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
    
    # Create directories for the app and datastore.
    os.makedirs(args.directory)
    os.makedirs(os.path.join(args.directory,'datastore'))
    
    dr = args.directory
    # get the zounds installation path.
    import zounds
    zp = os.path.join(zounds.__path__[0],'quickstart')
    # copy the files to the new app directory
    for fn in os.listdir(zp):
        if fn != os.path.split(__file__)[1]:
            shutil.copy(os.path.join(zp,fn),os.path.join(dr,fn))
    
    # Read the contents of the config file template
    configfile = os.path.join(dr,'config.py')
    with open(configfile,'r') as f:
        config_t = string.Template(f.read())
    
    # Substitute the user-specified source and directory parameters
    s = config_t.substitute({'Source'    : args.source,
                             'Directory' : args.directory})
    
    # Write the real config file to disk
    with open(configfile,'w') as f:
        f.write(s)
    
    
        
    
    
    