Installation:

`apt install python3-pip mencoder`
`python3 -m venv env
source env/bin/activate
python setup.py install`

Usage:
`python detect_blur.py  >file_list_for_mencoder.txt && mencoder "mf://@file_list_for_mencoder.txt" -mf fps=30:type=jpeg -noskip -of lavf -lavfopts format=mkv -ovc lavc -lavcopts vglobal=1:coder=0:vcodec=mjpeg -vf eq2=1.2:0.9:0.0:1.0,scale=4096:-2 -o <output file>`

