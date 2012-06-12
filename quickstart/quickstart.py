import os
import shutil
import argparse
import string

def copy(fn,dr):
    shutil.copy(fn,os.path.join(dr,fn))

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
    # copy the files to the new app directory
    copy('config.py',dr)
    copy('ingest.py',dr)
    copy('display.py',dr)
    copy('search.py',dr)
    copy('ab.py',dr)
    
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
    
    
        
    
    
    