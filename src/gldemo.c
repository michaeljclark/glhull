/*
 * gldemo - experiment to load and render béziergon glyphs
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
#include "nanovg_perf.h"

#include "linmath.h"
#include "gl2_util.h"
#include "cv_model.h"

static const char* dejavu_regular_fontpath = "fonts/DejaVuSans-Bold.ttf";
static const char* dejavu_bold_fontpath = "fonts/DejaVuSans-Bold.ttf";
static const char* curves_shader_glsl = "shaders/curves.comp";

typedef unsigned char uchar;

typedef struct hull_state hull_state;

struct hull_state
{
    GLFWwindow* window;
    NVGcontext* vg;
    cv_manifold *mb;
    uint glyph;
    uint contour;
    uint shape;
    uint point;
    int fontNormal;
    int fontBold;
};

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
    cv_load_face(state->mb, dejavu_bold_fontpath);
    state->glyph = cv_load_one_glyph(state->mb, 12, 100, 66);
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
    cv_debug("cv_draw_begin_path\n");
    nvgBeginPath(vg);
}

static void cv_draw_move_to(NVGcontext* vg, float x, float y)
{
    cv_debug("cv_draw_move_to: %f %f\n", x, y);
    nvgMoveTo(vg, x, y);
}

static void cv_draw_line_to(NVGcontext* vg, float x, float y)
{
    nvgLineTo(vg, x, y);
    cv_debug("cv_draw_line_to: %f, %f\n", x, y);
}

static void cv_draw_bezier_to(NVGcontext* vg, float c1x, float c1y,
    float c2x, float c2y, float x, float y)
{
    nvgBezierTo(vg, c1x, c1y, c2x, c2y, x, y);
    cv_debug("cv_draw_bezier_to: %f, %f, %f, %f, %f, %f\n",
        c1x, c1y, c2x, c2y, x, y);
}

static void cv_draw_quadratic_to(NVGcontext* vg, float cx, float cy,
    float x, float y)
{
    nvgQuadTo(vg, cx, cy, x, y);
    cv_debug("cv_draw_quadratic_to: %f, %f, %f, %f\n", cx, cy, x, y);
}

static void cv_draw_close_path(NVGcontext* vg)
{
    cv_debug("cv_draw_close_path\n");
    nvgClosePath(vg);
}

static void cv_draw_fill(NVGcontext* vg)
{
    cv_debug("cv_draw_fill\n");
    nvgFill(vg);
}

static void cv_draw_path_winding(NVGcontext* vg, int dir)
{
    cv_debug("cv_draw_path_winding: %d\n", dir);
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
    nvgRotate(vg, sinf(t*5.0f)*45.0f/180.0f*NVG_PI);
    nvgFillColor(vg, nvgRGBA(255,255,255,255));
    hull_traverse_reset(state);
    hull_traverse_nodes(state, o, s, glyph->shape, end);
    if (state->shape != -1) {
        hull_path_winding(state);
        cv_draw_fill(vg);
    }
    nvgRestore(vg);

    cv_ll = cv_ll_info;
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

int main()
{
    GLFWwindow* window;
    hull_state state;
    PerfGraph fps;
    double prevt = 0;

    if (!glfwInit()) {
        cv_panic("glfwInit failed\n");
    }

    initGraph(&fps, GRAPH_RENDER_FPS, "Frame Time");

    glfwSetErrorCallback(errorcb);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR , GL_TRUE);

    state.window = window = glfwCreateWindow(1000, 600, "gldemo", NULL, NULL);
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
    glClearColor(0.3f, 0.3f, 0.32f, 1.0f);

    hull_state_init(&state);
    prevt = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        double mx, my, t, dt;
        int winWidth, winHeight;
        int fbWidth, fbHeight;
        float pxRatio;

        t = glfwGetTime();
        dt = t - prevt;
        prevt = t;
        updateGraph(&fps, dt);

        glfwGetCursorPos(window, &mx, &my);
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        pxRatio = (float)fbWidth / (float)winWidth;

        glViewport(0, 0, fbWidth, fbHeight);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        nvgBeginFrame(state.vg, winWidth, winHeight, pxRatio);
        hull_render(&state, mx, my, winWidth,winHeight, t);
        renderGraph(state.vg, 5,5, &fps);
        nvgEndFrame(state.vg);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    hull_state_destroy(&state);
    glfwTerminate();

    return 0;
}
