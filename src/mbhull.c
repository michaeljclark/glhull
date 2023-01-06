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

static int opt_help;
static int opt_dump;
static int opt_glyph;
static int opt_glyph_s;
static int opt_glyph_e;
static int opt_rotate;
static int opt_trace;
static int opt_count;
static int opt_gpu;
static char* opt_polypath;
static char* opt_fontpath;
static char* opt_textpath;

typedef struct hull_state hull_state;
struct hull_state
{
    cv_manifold* mb;
    uint shape;
    uint contour;
    uint point;
    int max_edges;
    int edge_count;
    int fontNormal;
    int fontBold;
    int batch_cp;
    int batch_rot;
    FT_Library ftlib;
    FT_Face ftface;
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

/*
 * manifold compute shader
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
        cv_hull_rotate(state->mb, glyph, opt_rotate);
    }
}

void hull_graph_destroy(hull_state* state)
{
    cv_manifold_destroy(state->mb);
    free(state->mb);
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
        "  -bt, --batch-tmpl <txtfile>        write text\n"
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
            opt_polypath = argv[++i];
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

    if (!opt_fontpath && !opt_textpath) {
        cv_error("error: --font or --text option missing\n");
        opt_help++;
    }

    if (opt_help) {
        print_help(argc, argv);
        exit(1);
    }
}

static void hull_write_poly_header(hull_state *state, FILE *f,
    uint cp, uint rot, uint opts, uint nvertices, uint nfaces)
{
    cv_manifold *mb = state->mb;
    fprintf(f, "ply\n");
    fprintf(f, "format ascii 1.0\n");
    fprintf(f, "comment created by mbhull\n");
    fprintf(f, "comment codepoint %d rotation %d direction %s\n",
        cp, rot, opts == cv_hull_transform_forward ? "fwd" : "rev");
    fprintf(f, "element vertex %d\n", nvertices);
    fprintf(f, "property float32 x\n");
    fprintf(f, "property float32 y\n");
    fprintf(f, "element face %d\n", nfaces);
    fprintf(f, "property list uint8 int32 vertex_indices\n");
    fprintf(f, "end_header\n");
}

static void hull_write_poly_vertices(hull_state *state, vec2f *el, uint n,
    FILE *f)
{
    for (uint i = 0; i < n; i++) {
        fprintf(f, "%f %f\n", el[i].x, el[i].y);
    }
}

static void hull_write_poly_face(hull_state *state, vec2f *el, uint n,
    FILE *f, uint attr, cv_hull_range p, uint idx_offset)
{
    int s = p.s, e = (p.s <= p.e) ? p.e : p.e + n;

    if (e-s <= 1) { s = 0, e = n-1; }

    fprintf(f, "%d", e-s+1);
    switch(attr) {
    case cv_contour_cw:
        for (int i=e; i >= s; i--) fprintf(f, " %d", idx_offset + (i%n));
        break;
    case cv_contour_ccw:
        for (int i=s; i <= e; i++) fprintf(f, " %d", idx_offset + (i%n));
        break;
    }
    fprintf(f, "\n");
}

static int hull_transform_contours(hull_state *state, uint cidx, uint end, uint opts)
{
    cv_manifold *mb = state->mb;

    /* count contours */
    uint idx = cidx, ncontours = 0, nvertices = 0, contour;
    while (idx < end) {
        cv_node *node = cv_node_array_item(mb, idx);
        uint next = cv_node_next(node);
        ncontours++;
        idx = next ? next : end;
    }

    /* count vertices */
    uint* vcount = (uint*)alloca(sizeof(uint) * (ncontours+1));
    idx = cidx, contour = 0;
    vcount[contour++] = 0;
    while (idx < end) {
        cv_node *node = cv_node_array_item(mb, idx);
        uint n = cv_hull_edge_count(mb, idx, end);
        uint next = cv_node_next(node);
        nvertices += n;
        vcount[contour++] = nvertices;
        idx = next ? next : end;
    }

    FILE *f = stdout;
    if (opt_polypath) {
        static char filename[1024];
        snprintf(filename, sizeof(filename), opt_polypath, state->batch_cp,
            state->batch_rot, opts == cv_hull_transform_forward ? "fwd" : "rev");
        f = fopen(filename, "w");
        if (!f) cv_panic("batch: couldn't open file: %s\n", filename);
        cv_info("batch: writing poly: %s\n", filename);
    }

    /* write poly header */
    hull_write_poly_header(state, f, state->batch_cp, state->batch_rot, opts,
        nvertices, ncontours);

    /* write poly vertices */
    idx = cidx, contour = 0;
    while (idx < end) {
        CV_EDGE_LIST(mb,n,el,idx,end);
        cv_node *node = cv_node_array_item(mb, idx);
        uint next = cv_node_next(node);
        hull_write_poly_vertices(state, el, n, f);
        idx = next ? next : end;
    }

    /* write poly edges */
    idx = cidx, contour = 0;
    while (idx < end) {
        CV_EDGE_LIST(mb,n,el,idx,end);
        cv_node *node = cv_node_array_item(mb, idx);
        uint attr = cv_node_attr(node), next = cv_node_next(node);
        cv_hull_range p = cv_hull_split_contour(mb, el, n, idx, end, opts);
        hull_write_poly_face(state, el, n, f, attr, p, vcount[contour++]);
        idx = next ? next : end;
    }

    if (f != stdout) fclose(f);

    return 0;
}

static int hull_transform_shapes(hull_state *state, uint idx, uint end, uint opts)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(state->mb, idx);
        cv_trace("hull_transform_shapes: %s_%u\n", cv_node_type_name(node), idx);
        uint next = cv_node_next(node);
        uint contour_idx = idx + 1, contour_end = next ? next : end;
        if (contour_idx < contour_end) {
            hull_transform_contours(state, contour_idx, contour_end, opts);
        }
        idx = next ? next : end;
    }
    return 0;
}

static uint hull_transform(hull_state *state, uint idx, uint opts)
{
    cv_node *node = cv_node_array_item(state->mb, idx);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&state->mb->nodes);
    hull_transform_shapes(state, idx, end, opts);
}

static void hull_batch(hull_state *state)
{
    cv_hull_range p1, p2;

    for (int cp = opt_glyph_s; cp <= opt_glyph_e; cp++)
    {
        state->batch_cp = cp;
        uint glyph = cv_lookup_glyph(state->mb, cp);
        cv_glyph *g = cv_glyph_array_item(state->mb, glyph);
        int max_edges = hull_max_edges(state, g->shape);
        for (int r = 0; r < max_edges; r++)
        {
            state->batch_rot = r;
            hull_transform(state, g->shape, cv_hull_transform_forward);
            hull_transform(state, g->shape, cv_hull_transform_reverse);
            cv_hull_rotate(state->mb, g->shape, 1);
        }
    }
}

static void hull_main(hull_state *state)
{
    uint glyph = cv_lookup_glyph(state->mb, opt_glyph);
    cv_glyph *g = cv_glyph_array_item(state->mb, glyph);
    state->batch_cp = opt_glyph;
    state->batch_rot = opt_rotate;
    hull_transform(state, g->shape, opt_trace);
}

/*
 * main program
 */

static void mbhull_app(int argc, char **argv)
{
    GLFWwindow* window;
    hull_state state;

    memset(&state, 0, sizeof(state));
    hull_graph_init(&state);

    if (opt_count) {
        cv_dump_graph(state.mb);
        uint glyph = cv_lookup_glyph(state.mb, opt_glyph);
        cv_glyph *g = cv_glyph_array_item(state.mb, glyph);
        printf("%d\n", hull_max_edges(&state, g->shape));
        hull_graph_destroy(&state);
        exit(0);
    }

    glfwInit();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window = glfwCreateWindow(64, 64, "mbhull", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGL();

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

    if (opt_polypath) {
        hull_batch(&state);
    } else {
        hull_main(&state);
    }

    if (opt_gpu > 0) {
        cv_manifold_gpu mbo = { state.mb };
        cv_init_gpu(&mbo);
        cv_test_gpu(&mbo);
    }

    glfwTerminate();
}

int main(int argc, char **argv)
{
    cv_ll = cv_ll_debug;
    parse_options(argc, argv);
    mbhull_app(argc, argv);
    exit(0);
}
