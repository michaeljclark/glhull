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

typedef struct hull_state hull_state;

struct hull_state
{
    GLFWwindow* window;
    NVGcontext* vg;
    cv_manifold* mb;
    uint glyph;
    uint contour;
    uint shape;
    uint point;
    int fontNormal;
    int fontBold;
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

void hull_state_init(hull_state* state)
{
    state->vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
    if (!state->vg) {
        cv_panic("hull_state_init: error initializing nanovg\n");
    }

    state->fontNormal = nvgCreateFont(state->vg, "sans", dejavu_regular_fontpath);
    state->fontBold = nvgCreateFont(state->vg, "sans-bold", dejavu_bold_fontpath);

    state->mb = (cv_manifold*)calloc(1, sizeof(cv_manifold));
    cv_manifold_init(state->mb);
    cv_load_face(state->mb, opt_fontpath);
    state->glyph = cv_load_one_glyph(state->mb, 12, 100, opt_glyph);

    if (opt_rotate > 0) {
        cv_glyph *glyph = cv_glyph_array_item(state->mb, state->glyph);
        cv_hull_rotate(state->mb, glyph->shape, opt_rotate);
    }

    cv_dump_graph(state->mb);
}

void hull_state_destroy(hull_state* state)
{
    nvgDeleteGLES3(state->vg);
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

static int hull_convex_trace_contour(hull_state* state, vec2f o, float s,
    uint contour_idx, vec2f *edges, int n_edges, int start, int end, int dir, int w)
{
    cv_manifold* mb = state->mb;

    if (cv_ll <= cv_ll_debug)
    {
        cv_debug("hull: %-8s %-16s %-6s %-17s %-17s %-17s %-17s %-7s %-7s %-7s\n",
                 "id", "type", "split", "a(curr)",
                 "b(next)", "c(close)", "d(first)",
                 "ab", "bc", "cd");
        cv_debug("hull: %-8s %-16s %-6s %-17s %-17s %-17s %-17s %-7s %-7s %-7s\n",
                 "--------", "----------------", "------", "-----------------",
                 "-----------------", "-----------------", "-----------------",
                 "-------", "-------", "-------");
    }

    /* state variables */
    float ab_sl = (float)w;
    int split_at = -1;
    int edge_idx = contour_idx + 1;
    vec2f p0 = edges[start];
    vec2f p1 = edges[start + dir];
    vec2f d = (vec2f) { p1.x - p0.x, p1.y - p0.y };
    int i = start;

    /* find convex hull split point where winding order diverges */
    for (; i != end && split_at == -1; i += dir)
    {
        int i1 = i % n_edges;
        int i2 = (i + n_edges + dir) % n_edges;
        int i3 = (i + n_edges + dir + dir) % n_edges;

        vec2f v1 = edges[i1];
        vec2f v2 = edges[i2];
        vec2f v3 = edges[i3];

        vec2f a = (vec2f) { v2.x - v1.x, v2.y - v1.y };
        vec2f b = (vec2f) { v3.x - v2.x, v3.y - v2.y };
        vec2f c = (vec2f) { p0.x - v3.x, p0.y - v3.y };

        /* winding of current->next, next->closing, closing->first edges */
        float ab = vec2f_cross_z(a, b), ab_s = cv_extract_sign(ab);
        float bc = vec2f_cross_z(b, c), bc_s = cv_extract_sign(bc);
        float cd = vec2f_cross_z(c, d), cd_s = cv_extract_sign(cd);

        if (split_at == -1) {
            /* sign of ab changes, non-convex current->next edge */
            if (ab_s != 0.f && ab_sl != ab_s) {
                split_at = i + dir;
                cv_trace("hull: contour %u edge %u current->next edge winding "
                    "non convex\n", contour_idx, edge_idx + split_at);
            }
            /* sign of bc changes, non-convex next->closing edge */
            else if (bc_s != 0.f && ab_sl != bc_s && i != start) {
                split_at = i + dir;
                cv_trace("hull: contour %u edge %u next->closing edge winding "
                    "non convex\n", contour_idx, edge_idx + split_at);
            }
            /* sign of cd changes, non-convex closing->first edge */
            else if (cd_s != 0.f && ab_sl != cd_s && i != start) {
                split_at = i + dir;
                cv_trace("hull: contour %u edge %u closing->first edge winding "
                    "non convex\n", contour_idx, edge_idx + split_at);
            }
        }

        if (cv_ll <= cv_ll_debug)
        {
            cv_node *edge_node = cv_node_array_item(mb, edge_idx + i);
            cv_debug("hull: [%-6u] %-16s %-6s (%7.3f,%7.3f) (%7.3f,%7.3f) "
                "(%7.3f,%7.3f) (%7.3f,%7.3f) %7.1f %7.1f %7.1f\n",
                edge_idx + i, cv_node_type_name(edge_node), "",
                a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, ab, bc, cd);
        }

        ab_sl = ab_s;
    }

    /* shrink hull so it is not intersected by subsequent edges */
    for (; i != end && split_at != -1; i += dir)
    {
        int i1 = i % n_edges;
        int i2 = (i + n_edges + dir) % n_edges;
        int i3 = (i + n_edges + dir + dir) % n_edges;

        vec2f v1 = edges[i1];
        vec2f v2 = edges[i2];
        vec2f v3 = edges[i3];

        vec2f a = (vec2f) { v2.x - v1.x, v2.y - v1.y };
        vec2f b = (vec2f) { v3.x - v2.x, v3.y - v2.y };
        vec2f c = (vec2f) { p0.x - v3.x, p0.y - v3.y };

        /* winding of current->next, next->closing, closing->first edges */
        float ab = vec2f_cross_z(a, b), ab_s = cv_extract_sign(ab);
        float bc = vec2f_cross_z(b, c), bc_s = cv_extract_sign(bc);
        float cd = vec2f_cross_z(c, d), cd_s = cv_extract_sign(cd);

        /* walk backwards from the convex split point reducing the
         * size of the hull until the current point is not inside it */
        uint inside = 1;
        do {
            for (uint j = start; inside && j != split_at + dir; j += dir)
            {
                /* end point of the last edge is stored in the next
                 * edge so we must wrap around to the start edge */
                int i1 = j % n_edges;
                int i2 = (j + dir == split_at + dir ? start : j + dir) % n_edges;

                vec2f p1 = edges[i1 % n_edges];
                vec2f p2 = edges[i2 % n_edges];

                float d = vec2f_line_dist(p1, p2, v1);
                float b = cv_extract_sign(d);

                switch (w) {
                case -1: inside = b>0; break;
                case 1: inside = b<0; break;
                }
            }
            if (inside) {
                cv_trace("hull: contour edge %u inside %u:%u\n",
                         edge_idx + i, edge_idx, edge_idx + split_at);
            }
        } while (inside && (split_at -= dir) != start);

        if (cv_ll <= cv_ll_debug)
        {
            char str[16];
            snprintf(str, sizeof(str), "%-6d", edge_idx + split_at);
            cv_node *edge_node = cv_node_array_item(mb, edge_idx + i);
            cv_debug("hull: [%-6u] %-16s %-6s (%7.3f,%7.3f) (%7.3f,%7.3f) "
                "(%7.3f,%7.3f) (%7.3f,%7.3f) %7.1f %7.1f %7.1f\n",
                edge_idx + i, cv_node_type_name(edge_node), str,
                a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, ab, bc, cd);
        }
    }

    return split_at;
}

static float hull_px(vec2f o, float s, vec2f p) { return o.x + p.x * s; }
static float hull_py(vec2f o, float s, vec2f p) { return o.y + p.y * s; }

static int hull_convex_draw_contour(hull_state* state, vec2f o, float s,
    uint contour_idx, vec2f *edges, int n_edges, int start, int end,
    int dir, int w, int r)
{
    cv_manifold* mb = state->mb;
    NVGcontext* vg = state->vg;

    int split_at = r;
    int edge_idx = contour_idx + 1;
    vec2f p0 = edges[start];

    NVGcolor blue = nvgRGB(0x4e,0x79,0xa7);
    NVGcolor green = nvgRGB(0x8c,0xd1,0x7d);
    NVGcolor yellow = nvgRGBA(255,192,0,255);
    NVGcolor charcoal = nvgRGBA(32,32,32,255);

    nvgStrokeWidth(vg, 3.0f);
    nvgLineJoin(vg, NVG_ROUND);
    nvgLineCap(vg, NVG_ROUND);

    nvgBeginPath(vg);
    nvgMoveTo(vg, hull_px(o,s,p0), hull_py(o,s,p0));
    for (int i = start; i != end; i += dir)
    {
        int i2 = (i + n_edges + dir) % n_edges;
        vec2f v2 = edges[i2];
        nvgLineTo(vg, hull_px(o,s,v2), hull_py(o,s,v2));
    }
    nvgLineTo(vg, hull_px(o,s,p0), hull_py(o,s,p0));
    nvgStrokeColor(vg, blue);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgMoveTo(vg, hull_px(o,s,p0), hull_py(o,s,p0));
    for (int i = start; i != end && i != split_at; i += dir)
    {
        int i2 = (i + n_edges + dir) % n_edges;
        vec2f v2 = edges[i2];
        nvgLineTo(vg, hull_px(o,s,v2), hull_py(o,s,v2));
    }
    nvgLineTo(vg, hull_px(o,s,p0), hull_py(o,s,p0));
    nvgStrokeColor(vg, green);
    nvgStroke(vg);

    for (int i = start; i != end; i += dir)
    {
        int i1 = i % n_edges;
        int i2 = (i + n_edges + dir) % n_edges;
        int i3 = (i + n_edges + dir + dir) % n_edges;

        vec2f v1 = edges[i1];
        vec2f v2 = edges[i2];
        vec2f v3 = (vec2f) { (v1.x + v2.x) *0.5f, (v1.y + v2.y) *0.5f };

        char txt[16];
        float bounds[4];

        snprintf(txt, sizeof(txt), "%d", edge_idx + i);
        nvgFontSize(vg, 12.0f);
        nvgTextAlign(vg, NVG_ALIGN_RIGHT|NVG_ALIGN_MIDDLE);

        nvgTextBounds(vg, hull_px(o,s,v3), hull_py(o,s,v3), txt, NULL, bounds);

        nvgBeginPath(vg);
        nvgFillColor(vg, yellow);
        nvgRoundedRect(vg, (int)bounds[0]-4,
                           (int)bounds[1]-2,
                           (int)(bounds[2]-bounds[0])+8,
                           (int)(bounds[3]-bounds[1])+4,
                           ((int)(bounds[3]-bounds[1])+4)/2-1);
        nvgFill(vg);

        nvgFillColor(vg, charcoal);
        nvgText(vg, hull_px(o,s,v3), hull_py(o,s,v3), txt, NULL);
    }
}

static int hull_convex_transform_contour(hull_state* state, vec2f o, float s,
    uint idx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;
    cv_node *node = cv_node_array_item(mb, idx);
    uint next = cv_node_next(node);

    uint edge_idx = idx + 1, edge_end = next ? next : end;
    uint n_edges = edge_idx < edge_end ? edge_end - edge_idx : 0;
    vec2f *edges = (vec2f*)alloca(sizeof(vec2f) * n_edges);
    for (uint i = 0; i < n_edges; i++) {
        edges[i] = cv_edge_point(mb, edge_idx + i);
    }

    int w;
    switch(cv_node_attr(node)) {
    case cv_contour_cw: w = -1; break;
    case cv_contour_ccw: w = 1; break;
    default: w = 0; break;
    }

    int r;
    switch (opts) {
    case cv_hull_transform_auto:
        r = hull_convex_trace_contour(state, o, s, idx, edges, n_edges,
                                      0, n_edges, 1, w);
            hull_convex_draw_contour(state, o, s, idx, edges, n_edges,
                                     0, n_edges, 1, w, r);
        break;
    case cv_hull_transform_forward:
        r = hull_convex_trace_contour(state, o, s, idx, edges, n_edges,
                                      0, n_edges, 1, w);
            hull_convex_draw_contour(state, o, s, idx, edges, n_edges,
                                     0, n_edges, 1, w, r);
        break;
    case cv_hull_transform_reverse:
        r = hull_convex_trace_contour(state, o, s, idx, edges, n_edges,
                                      n_edges-1, -1, -1, -w);
            hull_convex_draw_contour(state, o, s, idx, edges, n_edges,
                                     n_edges-1, -1, -1, -w, r);
        break;
    }

    return r;
}

static int hull_convex_transform_contours(hull_state* state, vec2f o, float s,
    uint idx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;
    while (idx < end) {
        cv_node *node = cv_node_array_item(mb, idx);
        cv_debug("cv_hull_transform_contours: %s_%u\n",
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
        cv_debug("cv_hull_transform_shapes: %s_%u\n",
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
        "  -g, --glyph <int>                  character code\n"
        "  -c, --count                        count edges\n"
        "  -r, --rotate <int,int,int>         contour rotate\n"
        "  -t, --trace (auto|fwd|rev)         contour trace\n"
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
            if (strcmp(order, "auto") == 0) {
                opt_trace = cv_hull_transform_auto;
            } else if (strcmp(order, "fwd") == 0) {
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

    if (!opt_fontpath) {
        cv_error("error: --font option missing\n");
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

    if (opt_count) {
        cv_manifold *mb = (cv_manifold*)calloc(1, sizeof(cv_manifold));
        cv_manifold_init(mb);
        cv_load_face(mb, opt_fontpath);
        cv_load_one_glyph(mb, 12, 100, opt_glyph);
        cv_dump_graph(mb);
        printf("%d\n", hull_max_edges(mb));
        cv_manifold_destroy(mb);
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

    hull_state_init(&state);

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

    hull_state_destroy(&state);
    glfwTerminate();
}

/*
 * main program
 */

int main(int argc, char **argv)
{
    parse_options(argc, argv);
    glhull_app(argc, argv);
    return 0;
}
