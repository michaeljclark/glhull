/*
 * glhull - experiment to render béziergon convex interior hulls
 */

#include <stdio.h>
#include <string.h>
#include <alloca.h>
#include <float.h>
#include <errno.h>
#include <sys/stat.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_MODULE_H
#include FT_GLYPH_H
#include FT_OUTLINE_H

#include "nanovg.h"
#define NANOVG_GLES3_IMPLEMENTATION
#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "linmath.h"
#include "gl2_util.h"
#include "cv_model.h"

typedef unsigned char uchar;

static const char* dejavu_regular_fontpath = "fonts/DejaVuSans-Bold.ttf";
static const char* dejavu_bold_fontpath = "fonts/DejaVuSans-Bold.ttf";
static const char* curves_shader_glsl = "shaders/curves.comp";

static int opt_help;
static int opt_glyph;
static int opt_rotate;
static int opt_trace;
static int opt_count;
static char* opt_imagepath;
static char* opt_fontpath;
static char* opt_textpath;

typedef struct hull_state hull_state;
struct hull_state
{
    GLFWwindow* window;
    NVGcontext* vg;
    cv_manifold* mb;
    uint glyph;
    uint shape;
    uint contour;
    uint point;
    int fontNormal;
    int fontBold;
    FT_Library ftlib;
    FT_Face ftface;
};

static int max_edges = 0;
static int edge_count = 0;

static int hull_count_edges(cv_manifold *ctx, uint idx, uint end, uint depth, void *userdata)
{
    cv_node *node = cv_node_array_item(ctx, idx);
    uint type = cv_node_type(node);
    switch (type) {
    case cv_type_2d_shape:
        break;
    case cv_type_2d_contour:
        max_edges = cv_max(max_edges, edge_count);
        edge_count = 0;
        break;
    case cv_type_2d_edge_linear:
    case cv_type_2d_edge_conic:
    case cv_type_2d_edge_cubic:
        edge_count++;
        break;
    }
    return 1;
}

static uint hull_max_edges(cv_manifold *ctx)
{
    uint end = array_buffer_count(&ctx->nodes);
    cv_traverse_nodes(ctx, 0, end, 0, 0, hull_count_edges);
    max_edges = cv_max(max_edges, edge_count);
    return max_edges;
}

static void hull_vg_init(hull_state* state)
{
    state->vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
    if (!state->vg) {
        cv_panic("hull_state_init: error initializing nanovg\n");
    }

    state->fontNormal = nvgCreateFont(state->vg, "sans", dejavu_regular_fontpath);
    state->fontBold = nvgCreateFont(state->vg, "sans-bold", dejavu_bold_fontpath);
}

static void hull_graph_init(hull_state* state)
{
    FT_Error fterr;

    state->mb = (cv_manifold*)calloc(1, sizeof(cv_manifold));
    cv_manifold_init(state->mb);

    if (opt_fontpath) {
        state->ftlib = cv_init_ftlib();
        state->ftface = cv_load_ftface(state->ftlib, opt_fontpath);
        state->glyph = cv_load_ftglyph(state->mb, state->ftface, 12, 100, opt_glyph);
    } else if (opt_textpath) {
        state->glyph = cv_load_glyph_text_file(state->mb, opt_textpath);
    }

    if (opt_rotate > 0) {
        cv_glyph *glyph = cv_glyph_array_item(state->mb, state->glyph);
        cv_hull_rotate(state->mb, glyph->shape, opt_rotate);
    }

    cv_dump_graph(state->mb);
}

void hull_vg_destroy(hull_state* state)
{
    nvgDeleteGLES3(state->vg);
}

void hull_graph_destroy(hull_state* state)
{
    cv_manifold_destroy(state->mb);
    free(state->mb);
}

static void cv_draw_begin_path(NVGcontext* vg)
{
    cv_debug("trace: cv_draw_begin_path\n");
    nvgBeginPath(vg);
}

static void cv_draw_move_to(NVGcontext* vg, float x, float y)
{
    cv_debug("trace: cv_draw_move_to: %f %f\n", x, y);
    nvgMoveTo(vg, x, y);
}

static void cv_draw_line_to(NVGcontext* vg, float x, float y)
{
    nvgLineTo(vg, x, y);
    cv_debug("trace: cv_draw_line_to: %f, %f\n", x, y);
}

static void cv_draw_bezier_to(NVGcontext* vg, float c1x, float c1y,
    float c2x, float c2y, float x, float y)
{
    nvgBezierTo(vg, c1x, c1y, c2x, c2y, x, y);
    cv_debug("trace: cv_draw_bezier_to: %f, %f, %f, %f, %f, %f\n",
        c1x, c1y, c2x, c2y, x, y);
}

static void cv_draw_quadratic_to(NVGcontext* vg, float cx, float cy,
    float x, float y)
{
    nvgQuadTo(vg, cx, cy, x, y);
    cv_debug("trace: cv_draw_quadratic_to: %f, %f, %f, %f\n", cx, cy, x, y);
}

static void cv_draw_close_path(NVGcontext* vg)
{
    cv_debug("trace: cv_draw_close_path\n");
    nvgClosePath(vg);
}

static void cv_draw_fill(NVGcontext* vg)
{
    cv_debug("trace: cv_draw_fill\n");
    nvgFill(vg);
}

static void cv_draw_path_winding(NVGcontext* vg, int dir)
{
    cv_debug("trace: cv_draw_path_winding: %d\n", dir);
    nvgPathWinding(vg, dir);
}

static void hull_path_winding(hull_state *state)
{
    NVGcontext* vg = state->vg;
    cv_manifold* cv = state->mb;
    if (state->contour != -1) {
        switch (cv_node_attr(cv_node_array_item(cv, state->contour))) {
        case cv_contour_ccw: cv_draw_path_winding(vg, NVG_SOLID); break;
        case cv_contour_cw: cv_draw_path_winding(vg, NVG_HOLE); break;
        default: break;
        }
    }
}

static void hull_traverse_reset(hull_state *state)
{
    state->shape = state->contour = state->point = -1;
}

static float hull_x(vec2f o, float s, cv_point *p) { return o.x + p->v.x * s; }
static float hull_y(vec2f o, float s, cv_point *p) { return o.y + p->v.y * s; }

static void hull_traverse_nodes(hull_state *state, vec2f o, float s,
    uint idx, uint end)
{
    NVGcontext* vg = state->vg;
    cv_manifold* cv = state->mb;
    while (idx < end) {
        cv_node *node = cv_node_array_item(cv, idx);
        uint next = cv_node_next(node);
        uint type = cv_node_type(node);
        uint offset = cv_node_offset(node);
        switch (type) {
            case cv_type_2d_edge_linear:
            case cv_type_2d_edge_conic:
            case cv_type_2d_edge_cubic: {
                if (state->point == -1) {
                    cv_point *p0 = cv_point_array_item(cv, offset + 0);
                    cv_draw_move_to(vg, hull_x(o, s, p0), hull_y(o, s, p0));
                    state->point = idx;
                }
            }
        }
        switch (type) {
            case cv_type_2d_shape: {
                if (state->shape != -1) {
                    cv_draw_fill(vg);
                }
                state->shape = idx;
                state->contour = -1;
                cv_draw_begin_path(vg);
                break;
            }
            case cv_type_2d_contour: {
                if (state->contour != -1) {
                    hull_path_winding(state);
                    cv_draw_close_path(vg);
                }
                state->contour = idx;
                state->point = -1;
                break;
            }
            case cv_type_2d_edge_linear: {
                cv_point *p1 = cv_point_array_item(cv, offset + 1);
                cv_draw_line_to(vg, hull_x(o, s, p1), hull_y(o, s, p1));
                break;
            }
            case cv_type_2d_edge_conic: {
                cv_point *p1 = cv_point_array_item(cv, offset + 1);
                cv_point *p2 = cv_point_array_item(cv, offset + 2);
                cv_draw_quadratic_to(vg, hull_x(o, s, p1), hull_y(o, s, p1),
                                    hull_x(o, s, p2), hull_y(o, s, p2));
                break;
            }
            case cv_type_2d_edge_cubic: {
                cv_point *p1 = cv_point_array_item(cv, offset + 1);
                cv_point *p2 = cv_point_array_item(cv, offset + 2);
                cv_point *p3 = cv_point_array_item(cv, offset + 3);
                cv_draw_bezier_to(vg, hull_x(o, s, p1), hull_y(o, s, p1),
                                 hull_x(o, s, p2), hull_y(o, s, p2),
                                 hull_x(o, s, p3), hull_y(o, s, p3));
                break;
            }
        }
        uint new_end = next ? next : end;
        if (idx + 1 < new_end) {
            switch (type) {
            case cv_type_2d_shape:
            case cv_type_2d_contour:
                hull_traverse_nodes(state, o, s, idx + 1, new_end);
                break;
            }
        }
        idx = next ? next : end;
    }
}

static float hull_px(vec2f o, float s, vec2f p) { return o.x + p.x * s; }
static float hull_py(vec2f o, float s, vec2f p) { return o.y + p.y * s; }

typedef struct hull_draw_state hull_draw_state;
struct hull_draw_state
{
    NVGcontext* vg;
    cv_manifold* mb;
    vec2f o;
    float x;
};

static int hull_convex_draw_contour(hull_state *state, hull_draw_state ds,
    uint idx, vec2f *edges, int n, int s, int e, int dir, int r)
{
    cv_manifold *mb = state->mb;
    NVGcontext* vg = state->vg;

    vec2f o = ds.o;
    float x = ds.x;
    int split_idx = r;
    int edge_idx = idx + 1;
    int j0 = (s + n) % n;
    vec2f p0 = edges[j0];

    NVGcolor blue = nvgRGB(0x1f,0x77,0xb4);
    NVGcolor orange = nvgRGB(0xff,0x7f,0x0e);
    NVGcolor green = nvgRGB(0x2c,0xa0,0x2c);
    NVGcolor red = nvgRGB(0xd6,0x27,0x28);
    NVGcolor purple = nvgRGB(0x94,0x67,0xbd);
    NVGcolor brown = nvgRGB(0x8c,0x56,0x4b);
    NVGcolor pink = nvgRGB(0xe3,0x77,0xc2);
    NVGcolor grey = nvgRGB(0x7f,0x7f,0x7f);
    NVGcolor olive = nvgRGB(0xbc,0xbd,0x22);
    NVGcolor turquoise = nvgRGB(0x17,0xbe,0xcf);
    NVGcolor yellow = nvgRGB(0xff,0xc0,0x00);
    NVGcolor charcoal = nvgRGB(0x20,0x20,0x20);

    nvgStrokeWidth(vg, 3.0f);
    nvgLineJoin(vg, NVG_ROUND);
    nvgLineCap(vg, NVG_ROUND);

    nvgBeginPath(vg);
    nvgMoveTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    for (int i = s; i != e; i += dir)
    {
        int i2 = (i + n + dir) % n;
        vec2f v2 = edges[i2];
        nvgLineTo(vg, hull_px(o,x,v2), hull_py(o,x,v2));
    }
    nvgLineTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    nvgStrokeColor(vg, blue);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgMoveTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    for (int i = s; i != e && i != split_idx; i += dir)
    {
        int i2 = (i + n + dir) % n;
        vec2f v2 = edges[i2];
        nvgLineTo(vg, hull_px(o,x,v2), hull_py(o,x,v2));
    }
    nvgLineTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    nvgStrokeColor(vg, green);
    nvgStroke(vg);

    for (int i = s; i != e; i += dir)
    {
        int i1 = (i + n) % n;
        int i2 = (i + n + dir) % n;
        int i3 = (i + n + dir + dir) % n;

        vec2f v1 = edges[i1];
        vec2f v2 = edges[i2];
        vec2f v3 = (vec2f) { (v1.x + v2.x) *0.5f, (v1.y + v2.y) *0.5f };

        char txt[16];
        float bounds[4];

        snprintf(txt, sizeof(txt), "%d", edge_idx + (dir == 1 ? i1 : i2));
        nvgFontSize(vg, 12.0f);
        nvgTextAlign(vg, NVG_ALIGN_RIGHT|NVG_ALIGN_MIDDLE);

        nvgTextBounds(vg, hull_px(o,x,v3), hull_py(o,x,v3), txt, NULL, bounds);

        nvgBeginPath(vg);
        nvgFillColor(vg, (i0 % n) == 0 ? pink : yellow);
        nvgRoundedRect(vg, (int)bounds[0]-4,
                           (int)bounds[1]-2,
                           (int)(bounds[2]-bounds[0])+8,
                           (int)(bounds[3]-bounds[1])+4,
                           ((int)(bounds[3]-bounds[1])+4)/2-1);
        nvgFill(vg);

        nvgFillColor(vg, charcoal);
        nvgText(vg, hull_px(o,x,v3), hull_py(o,x,v3), txt, NULL);
    }
}

typedef struct hull_draw_param hull_draw_param;
struct hull_draw_param { int s, e, dir, r; };

static int hull_convex_transform_contour(hull_state* state, vec2f o, float s,
    uint idx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;
    NVGcontext* vg = state->vg;

    cv_node *node = cv_node_array_item(mb, idx);
    uint next = cv_node_next(node);

    uint edge_idx = idx + 1, edge_end = next ? next : end;
    uint n = edge_idx < edge_end ? edge_end - edge_idx : 0;
    vec2f *el = (vec2f*)alloca(sizeof(vec2f) * n);
    for (uint i = 0; i < n; i++) {
        el[i] = cv_edge_point(mb, edge_idx + i);
    }

    int w;
    switch(cv_node_attr(node)) {
    case cv_contour_cw: w = -1; break;
    case cv_contour_ccw: w = 1; break;
    default: w = 0; break;
    }

    int r1, r2;
    hull_draw_state ds = { vg, mb, o, s };
    hull_draw_param dp;

    switch (opts) {
    case cv_hull_transform_forward:
        r1 = 0;
        do {
            /* skip concave edges and degenerate convex hulls */
            r2 = cv_hull_skip_contour(mb,idx,el,n,r1,n+r1,1,-w);
            r1 = cv_hull_trace_contour(mb,idx,el,n,r2,r2,n+r2,1,w);
            cv_trace("hull: fwd r1=%d r2=%d\n",
                cv_hull_edge(edge_idx, n, r1), cv_hull_edge(edge_idx, n, r2));
            dp = (hull_draw_param) { r2,r2+n,1,r1 };
        } while (r1-r2 < 2 && r1 != -1);
        if (r1 != -1) {
            /* attempt to expand hull in the opposite direction */
            r2 = cv_hull_trace_contour(mb,idx,el,n,n+r1,n+r2+1,r1,-1,-w);
            cv_trace("hull: fwd+rev r1=%d r2=%d\n",
                cv_hull_edge(edge_idx, n, r1), cv_hull_edge(edge_idx, n, r2));
            dp = (hull_draw_param) { r1+n,r1,-1,r2 };
        }
        hull_convex_draw_contour(state,ds,idx,el,n,dp.s,dp.e,dp.dir,dp.r);
        break;
    case cv_hull_transform_reverse:
        r1 = 0;
        do {
            /* skip concave edges and degenerate convex hulls */
            r2 = cv_hull_skip_contour(mb,idx,el,n,n+r1,r1,-1,w);
            r1 = cv_hull_trace_contour(mb,idx,el,n,n+r2,n+r2,r2,-1,-w);
            cv_trace("hull: rev r1=%d r2=%d\n",
                cv_hull_edge(edge_idx, n, r1), cv_hull_edge(edge_idx, n, r2));
            dp = (hull_draw_param) { r2+n,r2,-1,r1 };
        } while (r2+n-r1 < 2 && r1 != -1);
        if (r1 != -1) {
            /* attempt to expand hull in the opposite direction */
            r2 = cv_hull_trace_contour(mb,idx,el,n,r1,n+r2-1,n+r1,1,w);
            cv_trace("hull: rev+fwd r1=%d r2=%d\n",
                cv_hull_edge(edge_idx, n, r1), cv_hull_edge(edge_idx, n, r2));
            dp = (hull_draw_param) { r1,r1+n,1,r2 };
        }
        hull_convex_draw_contour(state,ds,idx,el,n,dp.s,dp.e,dp.dir,dp.r);
        break;
    }

    return r1;
}

static int hull_convex_transform_contours(hull_state* state, vec2f o, float s,
    uint idx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;
    while (idx < end) {
        cv_node *node = cv_node_array_item(mb, idx);
        cv_trace("hull_transform_contours: %s_%u\n",
            cv_node_type_name(node), idx);
        uint next = cv_node_next(node);
        hull_convex_transform_contour(state, o, s, idx, end, opts);
        idx = next ? next : end;
    }
    return 0;
}

static int hull_convex_transform_shapes(hull_state* state, vec2f o, float s,
    uint idx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;
    while (idx < end) {
        cv_node *node = cv_node_array_item(mb, idx);
        cv_trace("hull_transform_shapes: %s_%u\n",
            cv_node_type_name(node), idx);
        uint next = cv_node_next(node);
        uint contour_idx = idx + 1, contour_end = next ? next : end;
        if (contour_idx < contour_end) {
            hull_convex_transform_contours(state, o, s,
                contour_idx, contour_end, opts);
        }
        idx = next ? next : end;
    }
    return 0;
}

void hull_render(hull_state* state, float mx, float my, float w, float h, float t)
{
    NVGcontext* vg = state->vg;
    cv_manifold* cv = state->mb;

    cv_glyph *glyph = cv_glyph_array_item(cv, state->glyph);
    cv_node *node = cv_node_array_item(cv, glyph->shape);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&cv->nodes);

    float s = 30.0f;
    vec2f o = { - glyph->width/2 * s, + glyph->height/2 * s };

    nvgSave(vg);
    nvgTranslate(vg, w/2, h/2);
    nvgFillColor(vg, nvgRGB(255,255,255));
    hull_traverse_reset(state);
    hull_traverse_nodes(state, o, s, glyph->shape, end);
    if (state->shape != -1) {
        hull_path_winding(state);
        cv_draw_fill(vg);
    }
    hull_convex_transform_shapes(state, o, s, glyph->shape, end, opt_trace);
    nvgRestore(vg);

    cv_ll = cv_ll_info;
}

static void image_set_alpha(uchar* image, int w, int h, int stride, uchar a)
{
    int x, y;
    for (y = 0; y < h; y++) {
        uchar* row = &image[y*stride];
        for (x = 0; x < w; x++)
            row[x*4+3] = a;
    }
}

static void image_flip_horiz(uchar* image, int w, int h, int stride)
{
    int i = 0, j = h-1, k;
    while (i < j) {
        uchar* ri = &image[i * stride];
        uchar* rj = &image[j * stride];
        for (k = 0; k < w*4; k++) {
            uchar t = ri[k];
            ri[k] = rj[k];
            rj[k] = t;
        }
        i++;
        j--;
    }
}

void save_screenshot(int w, int h, const char* name)
{
    uchar* image = (uchar*)malloc(w*h*4);
    if (image == NULL) return;
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, image);
    image_set_alpha(image, w, h, w*4, 255);
    image_flip_horiz(image, w, h, w*4);
    stbi_write_png(name, w, h, 4, image, w*4);
    free(image);
}

void errorcb(int error, const char* desc)
{
    printf("GLFW error %d: %s\n", error, desc);
}

static void key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    NVG_NOTUSED(scancode);
    NVG_NOTUSED(mods);
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

/*
 * option processing
 */

static void print_help(int argc, char **argv)
{
    cv_info(
        "usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  -l, (info|debug|trace)             debug level\n"
        "  -f, --font <ttf>                   font file\n"
        "  -i, --text <contour>               text file\n"
        "  -g, --glyph <int>                  character code\n"
        "  -c, --count                        count edges\n"
        "  -r, --rotate <int,int,int>         contour rotate\n"
        "  -t, --trace (fwd|rev)              contour trace\n"
        "  -w, --write-image <pngfile>        write image\n"
        "  -h, --help                         command line help\n",
        argv[0]
    );
}

static int match_opt(const char *arg, const char *opt, const char *longopt)
{
    return strcmp(arg, opt) == 0 || strcmp(arg, longopt) == 0;
}

static void parse_options(int argc, char **argv)
{
    int i = 1;
    while (i < argc) {
        if (match_opt(argv[i], "-h", "--help")) {
            opt_help++;
            i++;
        } else if (match_opt(argv[i], "-l", "--level")) {
            char* level = argv[++i];
            if (strcmp(level, "none") == 0) {
                cv_ll = cv_ll_none;
            } else if (strcmp(level, "info") == 0) {
                cv_ll = cv_ll_info;
            } else if (strcmp(level, "debug") == 0) {
                cv_ll = cv_ll_debug;
            } else if (strcmp(level, "trace") == 0) {
                cv_ll = cv_ll_trace;
            }
            i++;
        } else if (match_opt(argv[i], "-f", "--font")) {
            opt_fontpath = argv[++i];
            i++;
        } else if (match_opt(argv[i], "-i", "--text")) {
            opt_textpath = argv[++i];
            i++;
        } else if (match_opt(argv[i], "-g", "--glyph")) {
            opt_glyph = atoi(argv[++i]);
            i++;
        } else if (match_opt(argv[i], "-c", "--count")) {
            opt_count = 1;
            i++;
        } else if (match_opt(argv[i], "-r", "--rotate")) {
            opt_rotate = atoi(argv[++i]);
            i++;
        } else if (match_opt(argv[i], "-t", "--trace")) {
            char* order = argv[++i];
            if (strcmp(order, "fwd") == 0) {
                opt_trace = cv_hull_transform_forward;
            } else if (strcmp(order, "rev") == 0) {
                opt_trace = cv_hull_transform_reverse;
            } else {
                cv_error("error: unknown --order option: %s\n", order);
                opt_help++;
            }
            i++;
        } else if (match_opt(argv[i], "-w", "--write-image")) {
            opt_imagepath = argv[++i];
            i++;
        } else {
            cv_error("error: unknown option: %s\n", argv[i]);
            opt_help++;
            break;
        }
    }

    if (!opt_fontpath && !opt_textpath) {
        cv_error("error: --font or --text option missing\n");
        opt_help++;
    }

    if (opt_help) {
        print_help(argc, argv);
        exit(1);
    }
}

void glhull_app(int argc, char **argv)
{
    GLFWwindow* window;
    hull_state state;

    memset(&state, 0, sizeof(state));
    hull_graph_init(&state);

    if (opt_count) {
        cv_dump_graph(state.mb);
        printf("%d\n", hull_max_edges(state.mb));
        hull_graph_destroy(&state);
        exit(0);
    }

    if (!glfwInit()) {
        cv_panic("glfwInit failed\n");
    }

    glfwSetErrorCallback(errorcb);

    if (opt_imagepath) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    } else {
        glfwWindowHint(GLFW_SCALE_TO_MONITOR , GL_TRUE);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    state.window = window = glfwCreateWindow(600, 600, "glhull", NULL, NULL);
    if (!window) {
        cv_panic("glfwCreateWindow failed\n");
    }

    glfwSetWindowUserPointer(window, &state);
    glfwSetKeyCallback(window, key);
    glfwMakeContextCurrent(window);
    gladLoadGL();
    glfwSwapInterval(0);
    glfwSetTime(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);

    hull_vg_init(&state);

    while (!glfwWindowShouldClose(window))
    {
        double mx, my, t, dt;
        int winWidth, winHeight;
        int fbWidth, fbHeight;
        float pxRatio;

        glfwGetCursorPos(window, &mx, &my);
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        pxRatio = (float)fbWidth / (float)winWidth;

        glViewport(0, 0, fbWidth, fbHeight);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        nvgBeginFrame(state.vg, winWidth, winHeight, pxRatio);
        hull_render(&state, mx, my, winWidth,winHeight, t);
        nvgEndFrame(state.vg);

        if (opt_imagepath) {
            save_screenshot(fbWidth, fbHeight, opt_imagepath);
            exit(0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    hull_vg_destroy(&state);
    hull_graph_destroy(&state);

    glfwTerminate();
}

/*
 * main program
 */

int main(int argc, char **argv)
{
    cv_ll = cv_ll_debug;
    parse_options(argc, argv);
    glhull_app(argc, argv);
    return 0;
}
