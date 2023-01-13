#!/usr/bin/env python3

import os
import sys
import time
import argparse
import subprocess
import collections
import multiprocessing

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
parser.add_argument('-f', '--font', default=None,
                    help='font file')
parser.add_argument("-d", "--directory", default='fonts',
                    help="directory to search for fonts")
parser.add_argument("--extension", default='ttf',
                    help="file extension pattern")
parser.add_argument('-o', '--output', default='tmp/out',
                    help='output directory')
parser.add_argument('-s', '--start', default=33, type=int,
                    help='start codepoint')
parser.add_argument('-e', '--end', default=126, type=int,
                    help='end codepoint')
parser.add_argument('--progress', default=False, action='store_true',
                    help='print progress during batch')
args = parser.parse_args()

def basename_noext(font):
    return os.path.splitext(os.path.basename(font))[0]

def image_batch(tmpl, i, j, font):
    cmd = [ './build/glhull',
        '--font', font,
        '--level', str(args.level),
        '--glyph-range', '%d:%d' % (i, j),
        '--batch-tmpl', tmpl ]
    subprocess.run(cmd, check=True)

def poly_batch(tmpl, i, j, font):
    cmd = [ './build/mbhull',
        '--font', font,
        '--level', str(args.level),
        '--glyph-range', '%d:%d' % (i, j),
        '--batch-tmpl', tmpl ]
    subprocess.run(cmd, check=True)

def count_rotations(c, font):
    cmd = [ './build/glhull',
        '--font', font,
        '--level', 'none',
        '--glyph', str(c),
        '--count' ]
    return int(subprocess.check_output(cmd))

# global font_proc, num_fonts, num_glyphs, num_rotations, num_failed

def check_row(tmpl, s, c, k, dir, font):
    num_pass = 0
    num_fail = 0
    for r in range(0, k):
        poly_file = tmpl % (c, r, dir)
        if not os.path.exists(poly_file):
            continue
        ply = Polygon(poly_file)
        if ply.check():
            num_pass = num_pass + 1
        else:
            print('font %s non convex codepoint %d rotation %d direction %s'
                % (basename_noext(font), c, r, dir))
            num_fail = num_fail + 1
        os.unlink(poly_file)
    return num_pass, num_fail

def check_range(tmpl, s, i, j, font):
    num_glyph = 0
    num_pass = 0
    num_fail = 0
    for c in range(i, j+1):
        k = count_rotations(c, font)
        fwd_pass, fwd_fail = check_row(tmpl, s, c, k, 'fwd', font)
        rev_pass, rev_fail = check_row(tmpl, s, c, k, 'rev', font)
        num_glyph = num_glyph + 1
        num_pass = num_pass + fwd_pass + rev_pass
        num_fail = num_fail + fwd_fail + rev_fail
    return num_glyph, num_pass, num_fail

def print_row(f, s, c, k, dir):
    print("<tr>", file=f)
    print("<td>%s</td>" % (str(c)), file=f)
    for r in range(0, k):
        image_file = "hull_%d_%d_%s.png" % (c, r, dir)
        print("<td><img width=\"%d\" height=\"%d\" src=\"%s\"/></td>"
            % (s, s, image_file), file=f)
    print("</tr>", file=f)

def print_range(f, s, i, j, font):
    for c in range(i, j+1):
        k = count_rotations(c, font)
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

def html_batch(html_file, i, j, font):
    with open(html_file, 'w') as f:
        print_header(f, basename_noext(font))
        print_range(f, 128, i, j, font)
        print_footer(f)

file_list = []

if args.font is not None:
    file_list.append(args.font)
if args.directory:
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if filename.endswith(args.extension):
                file_list.append(os.path.join(dirpath, filename))

def fmt_ns(n):
    return "%d:%02d:%02d" % (int(n/1e9)/3600, (int(n/1e9)/60)%60, int(n/1e9)%60)

def do_image_file(font):
    out_dir = "%s/%s" % (args.output, basename_noext(font))
    html_file = "%s/index.html" % (out_dir)
    image_tmpl = "%s/hull_%%d_%%d_%%s.png" % (out_dir)
    os.makedirs(out_dir, exist_ok = True)
    html_batch(html_file, args.start, args.end, font)
    image_batch(image_tmpl, args.start, args.end, font)
    if args.progress:
        print("%s" % (font))
    return True

def do_poly_file(font):
    out_dir = "%s/%s" % (args.output, basename_noext(font))
    poly_tmpl = "%s/hull_%%d_%%d_%%s.ply" % (out_dir)
    os.makedirs(out_dir, exist_ok = True)
    start_ns = time.time_ns()
    poly_batch(poly_tmpl, args.start, args.end, font)
    check_ns = time.time_ns()
    num_glyph, num_pass, num_fail = check_range(poly_tmpl, 128, args.start, args.end, font)
    end_ns = time.time_ns()
    total_ns = end_ns - start_ns
    batch_ns = check_ns - start_ns
    verify_ns = end_ns - check_ns
    if args.progress:
        print("(batch %5.3f secs, verify %5.3f secs) %s"
            % (batch_ns/1e9, verify_ns/1e9, font))
    return num_glyph, num_pass, num_fail

if args.type == Type.IMAGE:
    with multiprocessing.Pool(os.cpu_count()) as p:
        results = p.map(do_image_file, file_list)

if args.type == Type.POLY:
    with multiprocessing.Pool(os.cpu_count()) as p:
        results = p.map(do_poly_file, file_list)
        total_glyph = 0
        total_pass = 0
        total_fail = 0
        for num_glyph, num_pass, num_fail in results:
            total_glyph = total_glyph + num_glyph
            total_pass = total_pass + num_pass
            total_fail = total_fail + num_fail
        percent = ( float(total_pass)/float(total_pass+total_fail)*100.0
                    if total_pass+total_fail > 0 else 0 )
        print("fonts %d glyphs %d rotations %d failed %d (%12.9f%%)"
            % (len(results), total_glyph, total_pass+total_fail, total_fail, percent))
