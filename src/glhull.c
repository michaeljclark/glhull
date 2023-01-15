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

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

#define FPNG_IMAGE_WRITE_IMPLEMENTATION
#include "fpng_c.h"

#include "linmath.h"
#include "gl2_util.h"
#include "cv_model.h"

typedef unsigned char uchar;

static const char* dejavu_regular_fontpath = "fonts/DejaVuSans-Bold.ttf";
static const char* dejavu_bold_fontpath = "fonts/DejaVuSans-Bold.ttf";
static const char* curves_shader_glsl = "shaders/curves.comp";

enum { opt_dump_metrics = 0x1, opt_dump_stats = 0x2, opt_dump_graph = 0x4 };

static const float min_zoom = 2.0f, max_zoom = 256.0f;

static cv_log_level cv_ll_save;
static int cv_ll_oneshot = 1;

static int opt_help;
static int opt_dump;
static int opt_glyph;
static int opt_glyph_s;
static int opt_glyph_e;
static int opt_rotate;
static int opt_trace;
static int opt_count;
static int opt_edgelabels;
static float opt_zoom = 32.0f;
static int opt_width = 512;
static int opt_height = 512;
static char* opt_imagepath;
static char* opt_fontpath;
static char* opt_textpath;

typedef struct hull_state hull_state;
struct hull_state
{
    GLFWwindow* window;
    NVGcontext* vg;
    cv_manifold* mb;
    uint shape;
    uint contour;
    uint point;
    int max_edges;
    int edge_count;
    int fontNormal;
    int fontBold;
    FT_Library ftlib;
    FT_Face ftface;
    vec2f mouse;
    vec2f origin;
    float zoom;
    vec2f last_mouse;
    float last_zoom;
};

static int hull_count_edges(cv_manifold *ctx, uint idx, uint end, uint depth, void *userdata)
{
    hull_state *state = (hull_state*)userdata;
    cv_node *node = cv_node_array_item(ctx, idx);
    uint type = cv_node_type(node);
    switch (type) {
    case cv_type_2d_shape:
        break;
    case cv_type_2d_contour:
        state->max_edges = cv_max(state->max_edges, state->edge_count);
        state->edge_count = 0;
        break;
    case cv_type_2d_edge_linear:
    case cv_type_2d_edge_conic:
    case cv_type_2d_edge_cubic:
        state->edge_count++;
        break;
    }
    return 1;
}

static uint hull_max_edges(hull_state* state, uint idx)
{
    cv_node *node = cv_node_array_item(state->mb, idx);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&state->mb->nodes);
    state->max_edges = state->edge_count = 0;
    cv_traverse_nodes(state->mb, idx, end, 0, state, hull_count_edges);
    state->max_edges = cv_max(state->max_edges, state->edge_count);
    return state->max_edges;
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
        if (opt_glyph) {
            cv_load_ftglyph(state->mb, state->ftface, 12, 100, opt_glyph);
        }
        if (opt_glyph_s && opt_glyph_e) {
            for (int cp = opt_glyph_s; cp <= opt_glyph_e; cp++) {
                cv_load_ftglyph(state->mb, state->ftface, 12, 100, cp);
            }
        }
    } else if (opt_textpath) {
        cv_load_glyph_text_file(state->mb, opt_textpath, opt_glyph);
    }

    if (opt_rotate > 0) {
        uint glyph = cv_lookup_glyph(state->mb, opt_glyph);
        cv_glyph *g = cv_glyph_array_item(state->mb, glyph);
        cv_hull_rotate(state->mb, g->shape, opt_rotate);
    }
}

void hull_vg_destroy(hull_state* state)
{
    nvgDeleteGLES3(state->vg);
}

void hull_graph_destroy(hull_state* state)
{
    cv_manifold_destroy(state->mb);
    cv_done_ftface(state->ftface);
    cv_done_ftlib(state->ftlib);
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

static int hull_convex_draw_contour(hull_state *state, vec2f *el, uint n,
    uint idx, uint end, vec2f o, float x, cv_hull_range hr)
{
    cv_manifold *mb = state->mb;
    NVGcontext* vg = state->vg;

    uint edge_idx = idx + 1;

    int s, e, dir, split_idx;
    if (hr.s < hr.e) {
        s = hr.s, e = hr.s + n, split_idx = hr.e, dir = 1;
    } else {
        s = hr.e + n, e = hr.e, split_idx = hr.s, dir = -1;
    }

    int j0 = (s + n) % n;
    vec2f p0 = el[j0];

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
        int i1 = (i + n + dir) % n;
        vec2f v1 = el[i1];
        nvgLineTo(vg, hull_px(o,x,v1), hull_py(o,x,v1));
    }
    nvgLineTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    nvgStrokeColor(vg, blue);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgMoveTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    for (int i = s; i != e && i != split_idx; i += dir)
    {
        int i1 = (i + n + dir) % n;
        vec2f v1 = el[i1];
        nvgLineTo(vg, hull_px(o,x,v1), hull_py(o,x,v1));
    }
    nvgLineTo(vg, hull_px(o,x,p0), hull_py(o,x,p0));
    nvgStrokeColor(vg, green);
    nvgStroke(vg);

    for (int i = s; i != e; i += dir)
    {
        int i1 = (i + n) % n;
        int i2 = (i + n + dir) % n;
        uint i0 = (dir == 1 ? i1 : i2);
        vec2f v0 = el[i0];

        char txt[16];
        float bounds[4];

        if (opt_edgelabels) {
            snprintf(txt, sizeof(txt), "%d", edge_idx + i0);
        } else {
            snprintf(txt, sizeof(txt), "%d", i0);
        }
        nvgFontSize(vg, 12.0f);
        nvgTextAlign(vg, NVG_ALIGN_RIGHT|NVG_ALIGN_MIDDLE);
        nvgTextBounds(vg, hull_px(o,x,v0), hull_py(o,x,v0), txt, NULL, bounds);
        nvgBeginPath(vg);
        nvgFillColor(vg, (i0 % n) == 0 ? pink : yellow);
        nvgRoundedRect(vg, (int)bounds[0]-4,
                           (int)bounds[1]-2,
                           (int)(bounds[2]-bounds[0])+8,
                           (int)(bounds[3]-bounds[1])+4,
                           ((int)(bounds[3]-bounds[1])+4)/2-1);
        nvgFill(vg);
        nvgFillColor(vg, charcoal);
        nvgText(vg, hull_px(o,x,v0), hull_py(o,x,v0), txt, NULL);
    }
}

static int hull_convex_transform_contours(hull_state* state, vec2f o, float s,
    uint idx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;
    while (idx < end) {
        CV_EDGE_LIST(mb,n,el,idx,end);
        cv_node *node = cv_node_array_item(mb, idx);
        cv_trace("hull_transform_contours: %s_%u\n", cv_node_type_name(node), idx);
        cv_hull_range hr;
        switch(cv_node_attr(node)) {
        case cv_contour_cw:
        case cv_contour_ccw:
            hr = cv_hull_split_contour(state->mb, el, n, idx, end, opts);
            hull_convex_draw_contour(state, el, n, idx, end, o, s, hr);
            break;
        }
        uint next = cv_node_next(node);
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

void hull_render(hull_state* state, int cp, int trace, float w, float h)
{
    NVGcontext* vg = state->vg;
    cv_manifold* cv = state->mb;

    uint glyph = cv_lookup_glyph(cv, cp);
    cv_glyph *g = cv_glyph_array_item(cv, glyph);
    cv_node *node = cv_node_array_item(cv, g->shape);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&cv->nodes);

    float s = state->zoom;
    vec2f o = { state->origin.x - g->width * .5f * s,
                state->origin.y + g->height * .5f * s };

    nvgSave(vg);
    nvgTranslate(vg, w/2, h/2);
    nvgFillColor(vg, nvgRGB(255,255,255));
    hull_traverse_reset(state);
    hull_traverse_nodes(state, o, s, g->shape, end);
    if (state->shape != -1) {
        hull_path_winding(state);
        cv_draw_fill(vg);
    }
    hull_convex_transform_shapes(state, o, s, g->shape, end, trace);
    nvgRestore(vg);
}

static int keycode_to_char(int key, int mods)
{
    // We convert simple Ctrl and Shift modifiers into ASCII
    if (key >= GLFW_KEY_SPACE && key <= GLFW_KEY_EQUAL) {
        if (mods == 0) {
            return key - GLFW_KEY_SPACE + ' ';
        }
    }
    if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z) {
        // convert Shift <Key> into ASCII
        if (mods == GLFW_MOD_SHIFT) {
            return key - GLFW_KEY_A + 'A';
        }
        // convert plain <Key> into ASCII
        if (mods == 0) {
            return key - GLFW_KEY_A + 'a';
        }
    }
    if (key >= GLFW_KEY_LEFT_BRACKET && key < GLFW_KEY_GRAVE_ACCENT) {
        // convert plain <Key> into ASCII
        if (mods == GLFW_MOD_SHIFT) {
            return key - GLFW_KEY_LEFT_BRACKET + '{';
        }
        if (mods == 0) {
            return key - GLFW_KEY_LEFT_BRACKET + '[';
        }
    }
    // convert Shift <Key> for miscellaneous characters
    if (mods == GLFW_MOD_SHIFT) {
        switch (key) {
        case GLFW_KEY_0:          /* ' */ return ')';
        case GLFW_KEY_1:          /* ' */ return '!';
        case GLFW_KEY_2:          /* ' */ return '@';
        case GLFW_KEY_3:          /* ' */ return '#';
        case GLFW_KEY_4:          /* ' */ return '$';
        case GLFW_KEY_5:          /* ' */ return '%';
        case GLFW_KEY_6:          /* ' */ return '^';
        case GLFW_KEY_7:          /* ' */ return '&';
        case GLFW_KEY_8:          /* ' */ return '*';
        case GLFW_KEY_9:          /* ' */ return '(';
        case GLFW_KEY_APOSTROPHE: /* ' */ return '"';
        case GLFW_KEY_COMMA:      /* , */ return '<';
        case GLFW_KEY_MINUS:      /* - */ return '_';
        case GLFW_KEY_PERIOD:     /* . */ return '>';
        case GLFW_KEY_SLASH:      /* / */ return '?';
        case GLFW_KEY_SEMICOLON:  /* ; */ return ':';
        case GLFW_KEY_EQUAL:      /* = */ return '+';
        case GLFW_KEY_GRAVE_ACCENT: /* ` */ return '~';
        }
    }
    return 0;
}

static void key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    NVG_NOTUSED(scancode);
    NVG_NOTUSED(mods);

    hull_state *state = (hull_state*)glfwGetWindowUserPointer(window);
    cv_ll_oneshot++;

    int c;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    } else if ((c = keycode_to_char(key, mods)) != 0 && action == GLFW_PRESS) {
        opt_glyph = c;
    } else if (key == GLFW_KEY_D && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL) {
        switch (opt_trace) {
        case cv_hull_transform_reverse: opt_trace = cv_hull_transform_forward; break;
        case cv_hull_transform_forward: opt_trace = cv_hull_transform_reverse; break;
        }
    } else if (key == GLFW_KEY_E && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL) {
        opt_edgelabels = !opt_edgelabels;
    } else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS && mods == 0) {
        uint glyph = cv_lookup_glyph(state->mb, opt_glyph);
        cv_glyph *g = cv_glyph_array_item(state->mb, glyph);
        uint num_edges = hull_max_edges(state, g->shape);
        cv_hull_rotate(state->mb, g->shape, num_edges-1);
    } else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS && mods == 0) {
        uint glyph = cv_lookup_glyph(state->mb, opt_glyph);
        cv_glyph *g = cv_glyph_array_item(state->mb, glyph);
        cv_hull_rotate(state->mb, g->shape, 1);
    }
}

static void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    hull_state *state = (hull_state*)glfwGetWindowUserPointer(window);

    float quantum = state->zoom / 16.f;
    float ratio = 1.f + (float)quantum / (float)state->zoom;
    if (yoffset < 0. && state->zoom < max_zoom) {
        state->origin.x *= ratio;
        state->origin.y *= ratio;
        state->zoom += quantum;
    } else if (yoffset > 0. && state->zoom > min_zoom) {
        state->origin.x /= ratio;
        state->origin.y /= ratio;
        state->zoom -= quantum;
    }
}

static int mouse_left_drag;
static int mouse_right_drag;

static void mouse_button(GLFWwindow* window, int button, int action, int mods)
{
    hull_state *state = (hull_state*)glfwGetWindowUserPointer(window);

    switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
        mouse_left_drag = (action == GLFW_PRESS);
        state->last_mouse = state->mouse;
        state->last_zoom = state->zoom;
        break;
    case GLFW_MOUSE_BUTTON_RIGHT:
        mouse_right_drag = (action == GLFW_PRESS);
        state->last_mouse = state->mouse;
        state->last_zoom = state->zoom;
        break;
    }
}

static void cursor_position(GLFWwindow* window, double xpos, double ypos)
{
    hull_state *state = (hull_state*)glfwGetWindowUserPointer(window);

    state->mouse = (vec2f) { xpos, ypos };

    if (mouse_left_drag) {
        state->origin.x += state->mouse.x - state->last_mouse.x;
        state->origin.y += state->mouse.y - state->last_mouse.y;
        state->last_mouse = state->mouse;
    }
    if (mouse_right_drag) {
        float delta0 = state->mouse.x - state->last_mouse.x;
        float delta1 = state->mouse.y - state->last_mouse.y;
        float zoom = state->last_zoom * powf(65.0f/64.0f,(float)-delta1);
        if (zoom != state->zoom && zoom > min_zoom && zoom < max_zoom) {
            state->zoom = zoom;
            state->origin.x = (state->origin.x * (zoom / state->zoom));
            state->origin.y = (state->origin.y * (zoom / state->zoom));
        }
    }
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

static void hull_save_screenshot(int w, int h, const char* filename)
{
    uchar* image = (uchar*)malloc(w*h*4);
    if (image == NULL) return;
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, image);
    image_set_alpha(image, w, h, w*4, 255);
    image_flip_horiz(image, w, h, w*4);
#ifdef STB_IMAGE_WRITE_IMPLEMENTATION
    stbi_write_png(filename, w, h, 4, image, w*4);
#endif
#ifdef FPNG_IMAGE_WRITE_IMPLEMENTATION
    fpng_encode_image_to_file(filename, image, w, h, 4, 0);
#endif
    free(image);
}

static void hull_batch_render(GLFWwindow* window, hull_state *state, int trace,
    int cp, int r)
{
    int winWidth, winHeight, fbWidth, fbHeight;
    float pxRatio;

    static char filename[1024];
    snprintf(filename, sizeof(filename), opt_imagepath, cp, r,
        trace == cv_hull_transform_forward ? "fwd" : "rev");

    glfwGetWindowSize(window, &winWidth, &winHeight);
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    pxRatio = (float)fbWidth / (float)winWidth;

    glViewport(0, 0, fbWidth, fbHeight);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
    nvgBeginFrame(state->vg, winWidth, winHeight, pxRatio);
    hull_render(state, cp, trace, winWidth, winHeight);
    nvgEndFrame(state->vg);

    cv_info("batch: writing image: %s\n", filename);
    hull_save_screenshot(fbWidth, fbHeight, filename);
}

static void hull_batch_loop(GLFWwindow* window, hull_state *state)
{
    for (int cp = opt_glyph_s; cp <= opt_glyph_e; cp++)
    {
        uint glyph = cv_lookup_glyph(state->mb, cp);
        cv_glyph *g = cv_glyph_array_item(state->mb, glyph);
        int max_edges = hull_max_edges(state, g->shape);
        for (int r = 0; r < max_edges; r++)
        {
            hull_batch_render(window, state, cv_hull_transform_forward, cp, r);
            glfwSwapBuffers(window);
            hull_batch_render(window, state, cv_hull_transform_reverse, cp, r);
            glfwSwapBuffers(window);
            cv_hull_rotate(state->mb, g->shape, 1);
        }
    }
}

static void hull_main_loop(GLFWwindow* window, hull_state *state)
{
    while (!glfwWindowShouldClose(window))
    {
        int winWidth, winHeight;
        int fbWidth, fbHeight;
        float pxRatio;

        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        pxRatio = (float)fbWidth / (float)winWidth;

        glViewport(0, 0, fbWidth, fbHeight);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        cv_ll_save = cv_ll;
        cv_ll = cv_ll_oneshot ? cv_ll_save : cv_ll_none;
        cv_ll_oneshot = 0;

        nvgBeginFrame(state->vg, winWidth, winHeight, pxRatio);
        hull_render(state, opt_glyph, opt_trace, winWidth,winHeight);
        nvgEndFrame(state->vg);

        cv_ll = cv_ll_save;

        glfwSwapBuffers(window);
        glfwPollEvents();
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
        "  -gr, --glyph-range <int>:<int>     character range\n"
        "  -c, --count                        count edges\n"
        "  -r, --rotate <int,int,int>         contour rotate\n"
        "  -t, --trace (fwd|rev)              contour trace\n"
        "  -bt, --batch-tmpl <pngfile>        write image\n"
        "  -dm, --dump-metrics                dump metrics\n"
        "  -ds, --dump-stats                  dump stats\n"
        "  -dg, --dump-graph                  dump graph\n"
        "  -e, --edge-labels                  edge labels\n"
        "  -z, --zoom <float>                 buffer zoom\n"
        "  -w, --width <int>                  buffer width\n"
        "  -h, --height <int>                 buffer height\n"
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
        } else if (match_opt(argv[i], "-gr", "--glyph-range")) {
            const char* arg = argv[++i];
            const char* colon = strchr(arg, ':');
            if (colon) {
                opt_glyph_s = atoi(arg);
                opt_glyph_e = atoi(colon+1);
            } else {
                cv_error("error: missing colon: --glyph-range\n");
                opt_help++;
            }
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
        } else if (match_opt(argv[i], "-bt", "--batch-tmpl")) {
            opt_imagepath = argv[++i];
            i++;
        } else if (match_opt(argv[i], "-dm", "--dump-metrics")) {
            opt_dump |= opt_dump_metrics;
            i++;
        } else if (match_opt(argv[i], "-ds", "--dump-stats")) {
            opt_dump |= opt_dump_stats;
            i++;
        } else if (match_opt(argv[i], "-dg", "--dump-graph")) {
            opt_dump |= opt_dump_graph;
            i++;
        } else if (match_opt(argv[i], "-e", "--edge-labels")) {
            opt_edgelabels++;
            i++;
        } else if (match_opt(argv[i], "-z", "--zoom")) {
            opt_zoom = (float)atof(argv[++i]);
            i++;
        } else if (match_opt(argv[i], "-w", "--width")) {
            opt_width = atoi(argv[++i]);
            i++;
        } else if (match_opt(argv[i], "-h", "--height")) {
            opt_height = atoi(argv[++i]);
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

static void errorcb(int error, const char* desc)
{
    cv_error("GLFW error %d: %s\n", error, desc);
}

void glhull_app(int argc, char **argv)
{
    GLFWwindow* window;
    hull_state state;

    memset(&state, 0, sizeof(state));
    state.zoom = opt_zoom;
    hull_graph_init(&state);

    if (opt_count) {
        cv_dump_graph(state.mb);
        uint glyph = cv_lookup_glyph(state.mb, opt_glyph);
        cv_glyph *g = cv_glyph_array_item(state.mb, glyph);
        printf("%d\n", hull_max_edges(&state, g->shape));
        hull_graph_destroy(&state);
        exit(0);
    }

    if ((opt_dump & opt_dump_metrics) > 0) {
        uint glyph = cv_lookup_glyph(state.mb, opt_glyph);
        cv_dump_metrics(state.mb, glyph);
    }
    if ((opt_dump & opt_dump_stats) > 0) {
        cv_dump_stats(state.mb);
    }
    if ((opt_dump & opt_dump_graph) > 0) {
        cv_dump_graph(state.mb);
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

    state.window = window = glfwCreateWindow(opt_width, opt_height,
        "glhull", NULL, NULL);
    if (!window) {
        cv_panic("glfwCreateWindow failed\n");
    }

    glfwSetWindowUserPointer(window, &state);
    glfwSetKeyCallback(window, key);
    glfwSetScrollCallback(window, scroll);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, cursor_position);
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

    if (opt_imagepath) {
        hull_batch_loop(window, &state);
    } else {
        hull_main_loop(window, &state);
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
