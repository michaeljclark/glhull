#!/usr/bin/env python3

import os
import sys
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

class Type(Enum):
    IMAGE = 'image'
    POLY = 'poly'

    def __str__(self):
        return self.value

def cross2f_z(a,b):
    return a[0]*b[1] - b[0]*a[1];

class Polygon:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.vertex_props = []
        self.face_props = []
        with open(filename, 'r') as f:
            # Expect 'ply' on the first line
            if f.readline().strip() != 'ply':
                raise ValueError('Invalid Stanford Polygon file')

            # Read through file and parse 'format', 'comment', 'element', and 'property' tokens
            for line in f:
                line = line.strip().split()
                if line[0] == 'format':
                    # Check that the file is in ASCII format
                    if line[1] != 'ascii':
                        raise ValueError('Invalid file format')
                elif line[0] == 'comment':
                    # Ignore comments
                    pass
                elif line[0] == 'element':
                    state = line[1]
                    if state == 'vertex':
                        num_vertices = int(line[-1])
                    elif state == 'face':
                        num_faces = int(line[-1])
                elif line[0] == 'property':
                    if state == 'vertex':
                        self.vertex_props.append(line[1:])
                    elif state == 'face':
                        self.face_props.append(line[1:])
                elif line[0] == 'end_header':
                    # Done reading header, start reading vertex and face data
                    break

            for i in range(0,num_vertices):
                line = f.readline().strip().split()
                vertex = tuple(map(float, line[:len(self.vertex_props)]))
                self.vertices.append(vertex)
            for i in range(0,num_faces):
                line = f.readline().strip().split()
                nfaces = int(line[0])
                face = tuple(map(int, line[1:nfaces+1]))
                self.faces.append(face)

    def check(self):
        debug = False
        convex = True
        nfaces = len(self.faces)
        for f in range(0,nfaces):
            face = self.faces[f]
            nedges = len(face)
            if debug:
                print('face_%d nedges=%d' % (f, nedges))
            for i in range(0,nedges):
                i0 = face[(i+0)%nedges]
                i1 = face[(i+1)%nedges]
                i2 = face[(i+2)%nedges]
                v1 = self.vertices[i0]
                v2 = self.vertices[i1]
                v3 = self.vertices[i2]
                a = (v2[0] - v1[0], v2[1] - v1[1])
                b = (v3[0] - v2[0], v3[1] - v2[1])
                z = cross2f_z(a,b)
                if debug:
                    print("(%7.3f,%7.3f) (%7.3f,%7.3f) %7.3f" % (a[0], a[1], b[0], b[1], z))
                convex = convex if z >= 0 else False
        return convex

parser = argparse.ArgumentParser(description='runs glhull tests')
parser.add_argument('-l', '--level', default=Level.INFO,
                    type=Level, choices=list(Level))
parser.add_argument('-t', '--type', default=Type.IMAGE,
                    type=Type, choices=list(Type))
parser.add_argument('-f', '--font', default='fonts/DejaVuSans-Bold.ttf',
                    help='font file')
parser.add_argument('-o', '--output', default='tmp/out',
                    help='output directory')
parser.add_argument('-s', '--start', default=65, type=int,
                    help='start codepoint')
parser.add_argument('-e', '--end', default=90, type=int,
                    help='end codepoint')
args = parser.parse_args()

def image_batch(tmpl, i, j):
    cmd = [ './build/glhull',
        '--font', args.font,
        '--level', str(args.level),
        '--glyph-range', '%d:%d' % (i, j),
        '--batch-tmpl', tmpl ]
    subprocess.run(cmd, check=True)

def poly_batch(tmpl, i, j):
    cmd = [ './build/mbhull',
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

def check_row(f, s, c, k, dir):
    for r in range(0, k):
        poly_file = f % (c, r, dir)
        ply = Polygon(poly_file)
        if not ply.check():
            print('non convex: codepoint %d rotation %d direction %s' % (c, r, dir))

def check_range(f, s, i, j):
    for c in range(i, j+1):
        k = count_rotations(c)
        check_row(f, s, c, k, 'fwd')
        check_row(f, s, c, k, 'rev')

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

def html_batch(html_file, i, j):
    with open(html_file, 'w') as f:
        print_header(f, title(args.font))
        print_range(f, 128, i, j)
        print_footer(f)

os.makedirs(args.output, exist_ok = True)

html_file = "%s/index.html" % (args.output)
image_tmpl = "%s/hull_%%d_%%d_%%s.png" % (args.output)
poly_tmpl = "%s/hull_%%d_%%d_%%s.ply" % (args.output)

if args.type == Type.IMAGE:
    html_batch(html_file, args.start, args.end)
    image_batch(image_tmpl, args.start, args.end)

if args.type == Type.POLY:
    poly_batch(poly_tmpl, args.start, args.end)
    check_range(poly_tmpl, 128, args.start, args.end)
