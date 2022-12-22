/*
 * mbhull - experiment to trace béziergon convex interior hulls
 */

#include <stdlib.h>
#include <stdio.h>
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

#include "linmath.h"
#include "gl2_util.h"
#include "cv_model.h"

static const char* curves_shader_glsl = "shaders/curves.comp";

enum { opt_dump_metrics = 0x1, opt_dump_stats = 0x2, opt_dump_graph = 0x4 };

static int opt_help = 0;
static int opt_dump = 0;
static int opt_glyph = 0;
static int opt_rotate = 0;
static int opt_trace = 0;
static int opt_gpu = 0;
static char* opt_fontpath;

/*
 * manifold gpu structure
 */

typedef struct cv_buffer cv_buffer;
typedef struct cv_result cv_result;
struct cv_result { uint count; };
struct cv_buffer { void *data; size_t len; };

typedef struct cv_manifold_gpu cv_manifold_gpu;
struct cv_manifold_gpu
{
    cv_manifold *mb;
    cv_result result;
    GLuint bin_program;
    GLuint meta_ubo;
    GLuint points_ssbo;
    GLuint nodes_ssbo;
    GLuint results_ssbo;
};

/*
 * manifold compute shader
 */

static void cv_init_gpu(cv_manifold_gpu *mbo)
{
    GLuint csh = compile_shader(GL_COMPUTE_SHADER, curves_shader_glsl);
    mbo->bin_program = link_program(&csh, 1, NULL);
    glGenBuffers(1, &mbo->meta_ubo);
    glGenBuffers(1, &mbo->points_ssbo);
    glGenBuffers(1, &mbo->nodes_ssbo);
    glGenBuffers(1, &mbo->results_ssbo);
}

void cv_test_gpu(cv_manifold_gpu *mbo)
{
    cv_buffer points = { array_buffer_data(&mbo->mb->points),
                         array_buffer_size(&mbo->mb->points) };
    cv_buffer nodes = { array_buffer_data(&mbo->mb->nodes),
                        array_buffer_size(&mbo->mb->nodes) };
    cv_meta meta = { (int)array_buffer_count(&mbo->mb->points),
                     (int)array_buffer_count(&mbo->mb->nodes) };

    glBindBuffer(GL_UNIFORM_BUFFER, mbo->meta_ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(meta), &meta, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mbo->points_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, points.len, points.data, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mbo->nodes_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, nodes.len, nodes.data, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mbo->results_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(cv_result), &mbo->result, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(mbo->bin_program);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, mbo->meta_ubo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mbo->points_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mbo->nodes_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, mbo->results_ssbo);
    glDispatchCompute(1, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    cv_result *result_map = (cv_result *)glMapNamedBuffer(mbo->results_ssbo, GL_READ_WRITE);
    memcpy(&mbo->result, result_map, sizeof(cv_result));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    printf("\ntest_gpu: result=0x%x\n", mbo->result.count);
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
        "  -r, --rotate <int,int,int>         contour rotate\n"
        "  -t, --trace (auto|fwd|rev)         contour trace\n"
        "  -dm, --dump-metrics                dump metrics\n"
        "  -ds, --dump-stats                  dump stats\n"
        "  -dg, --dump-graph                  dump graph\n"
        "  -tg, --test-gpu                    test gpu\n"
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
        } else if (match_opt(argv[i], "-dm", "--dump-metrics")) {
            opt_dump |= opt_dump_metrics;
            i++;
        } else if (match_opt(argv[i], "-ds", "--dump-stats")) {
            opt_dump |= opt_dump_stats;
            i++;
        } else if (match_opt(argv[i], "-dg", "--dump-graph")) {
            opt_dump |= opt_dump_graph;
            i++;
        } else if (match_opt(argv[i], "-tg", "--test-gpu")) {
            opt_gpu++;
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

/*
 * main program
 */

static void glcurves(int argc, char **argv)
{
    GLFWwindow* window;

    parse_options(argc, argv);

    glfwInit();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window = glfwCreateWindow(64, 64, "mbhull", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGL();

    cv_manifold src;
    cv_manifold_init(&src);
    cv_load_face(&src, opt_fontpath);

    uint glyph = cv_load_one_glyph(&src, 96, 100, opt_glyph);
    cv_glyph *g = cv_glyph_array_item(&src, glyph);

    if (opt_rotate > 0) {
        cv_hull_rotate(&src, g->shape, opt_rotate);
    }
    if ((opt_dump & opt_dump_metrics) > 0) {
        puts("");
        cv_dump_metrics(&src, glyph);
    }
    if ((opt_dump & opt_dump_stats) > 0) {
        puts("");
        cv_dump_stats(&src);
    }
    if ((opt_dump & opt_dump_graph) > 0) {
        puts("");
        cv_dump_graph(&src);
    }
    if (opt_trace > 0) {
        puts("");
        cv_manifold dst;
        cv_manifold_init(&dst);
        cv_hull_transform(&src, &dst, g->shape, opt_trace);
        puts("");
    }
    if (opt_gpu > 0) {
        cv_manifold_gpu mbo = { &src };
        cv_init_gpu(&mbo);
        cv_test_gpu(&mbo);
    }

    glfwTerminate();
}

int main(int argc, char **argv)
{
    glcurves(argc, argv);
    exit(0);
}
