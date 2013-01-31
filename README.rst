Introduction
============

Welcome to BatID. This software will enable you to quickly and easily identify 
the species associated with bat calls recorded using Anabat or Songmeter bat 
detectors. The current version of BatID is 1.0.

In order to get started, please read and follow the installation instructions 
below.


Installation
============

BatID requires that your computer have a working installation of Python 2.x 
available, as well as several packages for scientific computing, including 
numpy, matplotlib, and skikit-learn v0.12.

To easily install everything that you need, simply follow these three steps:

1. If you are reading this on github.com, download the entire BatID package by 
   going to https://github.com/jkitzes/batid/zipball/master. This will download 
   a zip file, which you can double-click to expand. This folder contains the 
   BatID software.

2. Install the Enthought Free python distribution. Download and install 
   Enthought Free from http://www.enthought.com/products/epd_free.php (you'll 
   need to fill out the form at the bottom of the page, which will take you 
   directly to a download link). If you are running Linux, chose the 32-bit 
   version.

3. Install sklearn v0.12. If you are using Windows, go to 
   http://sourceforge.net/projects/scikit-learn/files/, download the file 
   scikit-learn-0.12.win32-py2.7.exe, and run it. If you are using a Mac, 
   simply go to the BatID folder that you have downloaded (the folder 
   containing this README file), where you will see a subfolder called 'src'. 
   Open this subfolder, and change the name of the folder 'xsklearn' to 
   'sklearn'.

Once you have completed these steps, you can double-click on the Run_BatID file 
(on Windows) or the Run_BatID_Mac file (on a Mac) to launch the application. 
The application itself runs in your web browser - however, all analysis and 
data remain on your own computer, and no information is sent across the 
internet at any time.

If you are on a Mac, it's possible that you will receive an error the first 
time you double-click on Run_BatID_Mac. If so, try again - the program should 
launch on the second try.


Author
======

This software was written by Justin Kitzes <jkitzes@berkeley.edu>. Please 
contact Justin with comments or questions.


License
=======

This software is distributed under the BSD-new license.

Copyright (c) 2012, Justin Kitzes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this 
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
