#!/usr/bin/env python3

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='runs glhull tests')
parser.add_argument('-d', '--debug', default=False, action='store_true',
                    help='enable crefl debug output')
parser.add_argument('-o', '--output', default='tmp/out',
                    help='reflection metadata output')
args = parser.parse_args()

def exec_test(c, r, dir):
    os.makedirs(args.output, exist_ok = True)
    image_file = "hull_%d_%d_%s.png" % (c, r, dir)
    image_path = "%s/%s" % (args.output, image_file)
    cmd = [ './build/glhull',
        '--font', 'fonts/DejaVuSans-Bold.ttf',
        '--level', 'none',
        '--trace', dir,
        '--glyph', str(c),
        '--rotate', str(r),
        '--write-image', image_path ]
    if args.debug:
        print("%s" % cmd)
    subprocess.run(cmd, check=True)
    return image_file

def count_rotations(c):
    cmd = [ './build/glhull',
        '--font', 'fonts/DejaVuSans-Bold.ttf',
        '--level', 'none',
        '--glyph', str(c),
        '--count' ]
    return int(subprocess.check_output(cmd))

def exec_char(f, s, c, r, dir):
    image_file = exec_test(c, r, dir)
    print("<td><img width=\"%d\" height=\"%d\" src=\"%s\"/></td>" % (s, s, image_file), file=f)

def exec_row(f, s, c, k, dir):
    print("<tr>", file=f)
    print("<td>%s</td>" % (str(c)), file=f)
    for r in range(0, k):
        exec_char(f, s, c, r, dir)
    print("</tr>", file=f)

def exec_range(f, s, i, j):
    for c in range(i, j):
        k = count_rotations(c)
        exec_row(f, s, c, k, 'fwd')
        exec_row(f, s, c, k, 'rev')

def html_header(f):
    print("<html>", file=f)
    print("<head><title>glhull tests</title></head>", file=f)
    print("<body>", file=f)
    print("<table>", file=f)

def html_footer(f):
    print("</table>", file=f)
    print("</body>", file=f)
    print("</html>", file=f)

html_file = "%s/index.html" % (args.output)
with open(html_file, 'w') as f:
    html_header(f)
    #exec_range(f,150,33,126)
    exec_range(f,150,65,91)
    html_footer(f)
