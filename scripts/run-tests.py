#!/usr/bin/env python3

import os
import argparse
import subprocess

from enum import Enum

class Level(Enum):
    NONE = 'none'
    INFO = 'info'
    DEBUG = 'debug'
    TRACE = 'trace'

    def __str__(self):
        return self.value

parser = argparse.ArgumentParser(description='runs glhull tests')
parser.add_argument('-l', '--level', default=Level.INFO,
                    type=Level, choices=list(Level))
parser.add_argument('-f', '--font', default='fonts/DejaVuSans-Bold.ttf',
                    help='font file')
parser.add_argument('-o', '--output', default='tmp/out',
                    help='output directory')
parser.add_argument('-s', '--start', default=65, type=int,
                    help='start codepoint')
parser.add_argument('-e', '--end', default=90, type=int,
                    help='end codepoint')
args = parser.parse_args()

def execute_batch(tmpl, i, j):
    cmd = [ './build/glhull',
        '--font', args.font,
        '--level', str(args.level),
        '--glyph-range', '%d:%d' % (i, j),
        '--batch-tmpl', tmpl ]
    subprocess.run(cmd, check=True)

def count_rotations(c):
    cmd = [ './build/glhull',
        '--font', args.font,
        '--level', 'none',
        '--glyph', str(c),
        '--count' ]
    return int(subprocess.check_output(cmd))

def print_row(f, s, c, k, dir):
    print("<tr>", file=f)
    print("<td>%s</td>" % (str(c)), file=f)
    for r in range(0, k):
        image_file = "hull_%d_%d_%s.png" % (c, r, dir)
        print("<td><img width=\"%d\" height=\"%d\" src=\"%s\"/></td>" % (s, s, image_file), file=f)
    print("</tr>", file=f)

def print_range(f, s, i, j):
    for c in range(i, j+1):
        k = count_rotations(c)
        print_row(f, s, c, k, 'fwd')
        print_row(f, s, c, k, 'rev')

def print_header(f,title):
    print("<html>", file=f)
    print("<head><title>%s</title></head>" % (title), file=f)
    print("<body>", file=f)
    print("<table>", file=f)

def print_footer(f):
    print("</table>", file=f)
    print("</body>", file=f)
    print("</html>", file=f)

def title(font):
    return os.path.splitext(os.path.basename(font))[0]

os.makedirs(args.output, exist_ok = True)
html_file = "%s/index.html" % (args.output)
image_tmpl = "%s/hull_%%d_%%d_%%s.png" % (args.output)
with open(html_file, 'w') as f:
    print_header(f, title(args.font))
    print_range(f, 128, args.start, args.end)
    print_footer(f)
execute_batch(image_tmpl, args.start, args.end)
