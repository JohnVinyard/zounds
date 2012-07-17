import os.path
import shutil
import sys

import string
from string import Template
from config import *
from zounds.visualize.plot import plot
from zounds.nputil import safe_log

HtmlTemplate = Template('''
<html>
    <head>
        <!DOCTYPE html>
        <link rel="stylesheet" href="http://code.jquery.com/ui/1.8.18/themes/base/jquery-ui.css" type="text/css" media="all" />
        <link rel="stylesheet" href="http://static.jquery.com/ui/css/demo-docs-theme/ui.theme.css" type="text/css" media="all" />
        <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
        <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.8.18/jquery-ui.min.js"></script>
        <script type="text/javascript">
            jQuery(function() {
                jQuery("#accordion").accordion({fillSpace : true});
            });
        </script>
    </head>
    <body>
        <div id="accordion">
            ${Content}
        </div>
    </body>
</html>
''')

FeatureTemplate = Template('''
<h3>${Key}</h3>
<div><img src="${ImageUrl}"/></div>
''')

StringTemplate = Template('''
<h3>${Key}</h3>
<div>${Value}</div>
''')

PatternTemplate = Template('''
<h2><a href="#">${PatternId}</a></h2>
<div>
    ${Features}
</div>
''')


if __name__ == '__main__':

    path = sys.argv[1]
    if os.path.exists(path):
        shutil.rmtree(path)
    
    c = FrameModel.controller()
    _ids = list(c.list_ids())
    features = FrameModel.stored_features()

    pattern_html = []
    # only list the first 10 ids
    for _id in _ids[:10]:
        print _id
        pattern_path = os.path.join(path,_id)
        os.makedirs(pattern_path)
        feature_html = []
        data = c[_id]
        for k,v in features.iteritems():
            if 'framen' == k:
                continue
            
            print '\t%s' % k
            if 'S' == c.get_dtype(k).kind:
                feature_html.append(StringTemplate.substitute(\
                                                Key = k, Value = data[k][0]))
                continue
            
            
            fn = os.path.join(pattern_path,'%s.png' % k)
            url = os.path.join(_id,'%s.png' % k)
            plot(data[k],fn)
            feature_html.append(FeatureTemplate.substitute(ImageUrl = url, Key = k))
            
        pattern_html.append(PatternTemplate.substitute(PatternId = _id, Features = string.join(feature_html,'')))

    html = HtmlTemplate.substitute(Content = string.join(pattern_html,''))
    with open(os.path.join(path,'index.htm'),'w') as f:
        f.write(html)

        
        