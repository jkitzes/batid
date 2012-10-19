#!/usr/bin/python

'''
Minimal web server for batid gui.
'''

from wsgiref.simple_server import make_server
import sys, os, subprocess
import webbrowser
from string import Template

import Tkinter, tkFileDialog
import urlparse
from collections import OrderedDict
from src.utils import write_params

# ----------------------------------------------------------------------------
# Declare variables 
# ----------------------------------------------------------------------------

localport = 8000

root_path = os.path.dirname(os.path.abspath(__file__))
gui_path = os.path.join(root_path, 'gui')
script_file = os.path.join(root_path, 'src', 'classify.py')

# ----------------------------------------------------------------------------
# Main functions to handle requests and serve pages
# ----------------------------------------------------------------------------

def handle_request(environ, start_response):
    '''Determine which function should handle request given page name.'''

    # Get name of page requested
    page_name = environ['PATH_INFO'][1:]
   
    # Call appropriate function to serve page, looking for special cases
    if page_name == 'css':
        return css(environ, start_response)
    elif page_name == 'img':
        return img(environ, start_response)
    elif page_name == 'classify':
        return classify(environ, start_response)
    elif page_name == 'run':
        return run(environ, start_response)
    else:
        return get_page(environ, start_response)


def get_page(environ, start_response, message1='', message2=''):
    ''' Returns page as string. '''

    page_template = \
    Template(open(os.path.join(gui_path,'page_template.htm')).read())

    # Start response as html page
    start_response('200 OK', [('Content-Type', 'text/html')])
    
    # Get name of page requested, replacing '' with index
    page_name = environ['PATH_INFO'][1:]
    if page_name == '':
        page_name = 'index'

    # Check if txt file with body for page exists, if so get as Template
    try:
        page_path = os.path.join(gui_path, page_name + '.txt')
        page_body = Template(open(page_path).read())

    except:  # If page not found, get error page
        page_path = os.path.join(gui_path, '404.txt')
        page_body = Template(open(page_path).read())

    # Substitute filepath first, then pagebody and localport
    page_body_f = str(page_body.safe_substitute(message1=message1, 
                                                message2=message2))
    return page_template.safe_substitute(pagebody=page_body_f, 
                                         localport=localport)


# ----------------------------------------------------------------------------
# Handle special page requests
# ----------------------------------------------------------------------------

def css(environ, start_response):
    '''Return style.css CSS file from gui_path directory.'''
    start_response('200 OK', [('Content-Type', 'text/css')])
    css_file = open(os.path.join(gui_path,'style.css')).read()
    return css_file


def img(environ, start_response):
    '''Return image file from gui_path directory.'''
    start_response('200 OK', [('Content-Type', 'image/png')])
    query_dict = OrderedDict(urlparse.parse_qsl(environ['QUERY_STRING']))
    image_file = open(os.path.join(gui_path,query_dict['name']),'rb').read()
    return image_file


def classify(environ, start_response):
    '''Classify page opens file_browser if needed, serves page.'''

    query_dict = OrderedDict(urlparse.parse_qsl(environ['QUERY_STRING']))

    
    if 'get_file' in query_dict.keys():
        message1 = file_browser()
    else:
        message1 = ''
    
    return get_page(environ, start_response, message1)


def run(environ, start_response):
    '''Run page kicks off analysis as subprocess, then serves page.'''

    query_dict = OrderedDict(urlparse.parse_qsl(environ['QUERY_STRING']))

    # Write parameter file, catching any errors
    try:
        output_dir, file_name = os.path.split(query_dict['filepath'])
        assert output_dir != '', 'Invalid Anabat file location'
        query_dict['outputdir'] = output_dir  # Store output_dir as param
        write_params(query_dict, output_dir)

    except Exception, e:
        print 'Error in analysis setup:'
        print str(e)
        message1 = '''<p>Unfortunately, there was an error in the analysis, and 
                   no or only partial results were generated. The error is 
                   given below for debugging purposes:
                   </p><p><pre>%s</pre></p>''' % str(e)
        return get_page(environ, start_response, message1)  # End here
   
    # Start script and wait for it to complete, catching any errors
    try:
        subprocess.check_output(['python', script_file], cwd=output_dir, 
                                shell=False, stderr=subprocess.STDOUT)
        message1 = '''Success! Your results can be found in the directory 
        containing your Anabat scan file, which is %s''' % output_dir

    except subprocess.CalledProcessError, e:
        print 'Error in analysis:'
        print str(e.output)
        message1 = ('''<p>Unfortunately, there was an error in the analysis, 
                    and no or only partial results were generated. The error is 
                    given below for debugging purposes: 
                    </p><p><pre>%s</pre></p>''' % str(e.output))

    return get_page(environ, start_response, message1)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def file_browser():
    '''
    Browse for file.
    wx: see http://www.devshed.com/c/a/Python/Dialogs-in-wxPython/5/
    'darwin' hack makes window come to front in Mac
    Still need to test if window brought to front in Windows.'''

    # Set up hidden root window
    root = Tkinter.Tk()
    root.withdraw()

    # Bring dialog to front in Windows
    root.wm_attributes("-topmost", 1)

    # Bring dialog to front in Mac
    if sys.platform == 'darwin':
        if sys.maxsize > 2**32:  # 64bit, Tkinter is X11
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set '''
                      '''frontmost of process "X11" to true' ''')
        else:  #32bit, Tkinter is Python
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set '''
                      '''frontmost of process "Python" to true' ''')

    # Get filename and return it
    filename = tkFileDialog.askopenfilename(parent=root)

    return filename


# ----------------------------------------------------------------------------
# Start server and run forever (if file run as script)
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    httpd = make_server('', localport, handle_request)
    print 'Starting local webserver on port ' + str(localport)
    webbrowser.open('http://localhost:{!s}'.format(localport))
    httpd.serve_forever()
