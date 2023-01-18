/*
 * logging
 */

typedef enum {
    cv_ll_trace, cv_ll_debug, cv_ll_info, cv_ll_error, cv_ll_none
} cv_log_level;

static cv_log_level cv_ll = cv_ll_info;

#define cv_panic(...) do { fprintf(stderr, __VA_ARGS__); exit(1); } while(0);
#define cv_trace(...) if (cv_ll <= cv_ll_trace) printf(__VA_ARGS__);
#define cv_debug(...) if (cv_ll <= cv_ll_debug) printf(__VA_ARGS__);
#define cv_info(...)  if (cv_ll <= cv_ll_info) printf(__VA_ARGS__);
#define cv_error(...)  if (cv_ll <= cv_ll_error) printf(__VA_ARGS__);

/*
 * manifold buffer object model
 */

#define cv_type_2d_shape       0
#define cv_type_2d_contour     1
#define cv_type_2d_edge_linear 2
#define cv_type_2d_edge_conic  3
#define cv_type_2d_edge_cubic  4

#define cv_contour_zero     0
#define cv_contour_ccw      1
#define cv_contour_cw       2

typedef struct cv_manifold cv_manifold;
typedef struct cv_meta cv_meta;
typedef struct cv_node cv_node;
typedef struct cv_point cv_point;
typedef struct cv_glyph cv_glyph;
typedef struct cv_code cv_code;

typedef const FT_Vector ft_vec;

struct cv_meta { uint num_points, num_nodes; };
struct cv_node { uint type_next, attr_offset; };
struct cv_point { vec2f v; };
struct cv_glyph { uint codepoint, shape; float size, width, height;
                  float offset_x, offset_y, advance_x, advance_y; };
struct cv_code { uint glyph; };

/*
 * simple math
 */

#define cv_min(a,b) (((a)<(b))?(a):(b))
#define cv_max(a,b) (((a)>(b))?(a):(b))

static float vec2f_length(vec2f v)
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

static float vec2f_cross(vec2f a, vec2f b)
{
    return a.x*b.y - b.x*a.y;
}

static float vec2f_dot(vec2f a, vec2f b)
{
    return a.x*b.x + a.y*b.y;
}

static float vec2f_line_dist(vec2f p1, vec2f p2, vec2f v)
{
    return (v.x - p1.x) * (p2.y - p1.y) - (v.y - p1.y) * (p2.x - p1.x);
}

/*
 * manifold buffer context
 */

struct cv_manifold
{
    uint shape;
    uint contour;
    uint edge;
    uint point;

    float contour_area;
    vec2f contour_min;
    vec2f contour_max;

    float shape_area;
    vec2f shape_min;
    vec2f shape_max;

    array_buffer nodes;
    array_buffer points;
    array_buffer glyphs;
    array_buffer codes;
};

static void cv_manifold_init(cv_manifold *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->shape = ctx->contour = ctx->edge = ctx->point = -1;
    array_buffer_init(&ctx->nodes, sizeof(cv_node), 16);
    array_buffer_init(&ctx->points, sizeof(cv_point), 16);
    array_buffer_init(&ctx->glyphs, sizeof(cv_glyph), 16);
    array_buffer_init(&ctx->codes, sizeof(cv_code), 16);
}

static void cv_manifold_destroy(cv_manifold *ctx)
{
    array_buffer_destroy(&ctx->nodes);
    array_buffer_destroy(&ctx->points);
    array_buffer_destroy(&ctx->glyphs);
}

/*
 * manifold buffer interface
 */

static cv_node* cv_node_array_item(cv_manifold *ctx, size_t idx)
{
    return (cv_node*)array_buffer_data(&ctx->nodes) + idx;
}

static cv_point* cv_point_array_item(cv_manifold *ctx, size_t idx)
{
    return (cv_point*)array_buffer_data(&ctx->points) + idx;
}

static cv_glyph* cv_glyph_array_item(cv_manifold *ctx, size_t idx)
{
    return (cv_glyph*)array_buffer_data(&ctx->glyphs) + idx;
}

static cv_code* cv_code_array_item(cv_manifold *ctx, size_t idx)
{
    return (cv_code*)array_buffer_data(&ctx->codes) + idx;
}

static uint cv_node_type(cv_node *node) { return node->type_next >> 28; }
static uint cv_node_next(cv_node *node) { return node->type_next << 4 >> 4; }
static uint cv_node_attr(cv_node *node) { return node->attr_offset >> 28; }
static uint cv_node_offset(cv_node *node) { return node->attr_offset << 4 >> 4; }

static uint cv_point_count(uint type)
{
    switch(type) {
    case cv_type_2d_shape: return 2; /* minmax */
    case cv_type_2d_contour: return 2; /* minmax */
    case cv_type_2d_edge_linear: return 2;
    case cv_type_2d_edge_conic: return 3;
    case cv_type_2d_edge_cubic: return 4;
    default: break;
    }
    return 0;
}

static const char* cv_node_type_name(cv_node *node)
{
    switch(cv_node_type(node)) {
    case cv_type_2d_shape: return "Shape2D";
    case cv_type_2d_contour: return "Contour2D";
    case cv_type_2d_edge_linear: return "EdgeLinear2D";
    case cv_type_2d_edge_conic: return "EdgeQuadratic2D";
    case cv_type_2d_edge_cubic: return "EdgeCubic2D";
    default: break;
    }
    return "Unknown";
}

static const char* cv_contour_attr_name(cv_node *node)
{
    switch(cv_node_attr(node)) {
    case cv_contour_cw: return "Hole";
    case cv_contour_ccw: return "Solid";
    default: break;
    }
    return "Unknown";
}

static void cv_node_set_type(cv_node *node, uint type)
{
    node->type_next = (type << 28) | (cv_node_next(node) & ((1u<<28)-1u));
}

static void cv_node_set_next(cv_node *node, uint next)
{
    node->type_next = (cv_node_type(node) << 28) | (next & ((1u<<28)-1u));
}

static void cv_node_set_attr(cv_node *node, uint attr)
{
    node->attr_offset = (attr << 28) | (cv_node_offset(node) & ((1u<<28)-1u));
}

static void cv_node_set_offset(cv_node *node, uint offset)
{
    node->attr_offset = (cv_node_attr(node) << 28) | (offset & ((1u<<28)-1u));
}

static uint cv_new_points(cv_manifold *ctx, uint count)
{
    uint idx = array_buffer_count(&ctx->points);
    cv_point p = {{ 0.f, 0.f }};
    for (uint i = 0; i < count; i++) {
        array_buffer_add(&ctx->points, &p);
    }
    return idx;
}

static uint cv_new_point(cv_manifold *ctx, float x, float y)
{
    uint idx = array_buffer_count(&ctx->points);
    cv_point p = {{ x, y }};
    array_buffer_add(&ctx->points, &p);
    return idx;
}

static void cv_finalize_contour(cv_manifold *ctx, int contour);
static void cv_finalize_shape(cv_manifold *ctx, int shape);

static uint cv_new_node(cv_manifold *ctx, int type, int offset)
{
    uint idx = array_buffer_count(&ctx->nodes);
    cv_node node = { (type << 28), offset };
    cv_point *p, *p1, *p2;
    size_t pc;

    array_buffer_add(&ctx->nodes, &node);

    switch (type) {
    case cv_type_2d_shape:
        if (ctx->shape < idx) {
            cv_node_set_next(cv_node_array_item(ctx, ctx->shape), idx);
            cv_finalize_shape(ctx, ctx->shape);
        }
        ctx->shape = idx;
        ctx->contour = -1;
        ctx->edge = -1;
        ctx->shape_area = 0.f;
        ctx->shape_min.x = FLT_MAX;
        ctx->shape_min.y = FLT_MAX;
        ctx->shape_max.x = -FLT_MAX;
        ctx->shape_max.y = -FLT_MAX;
        break;
    case cv_type_2d_contour:
        if (ctx->contour < idx) {
            cv_node_set_next(cv_node_array_item(ctx, ctx->contour), idx);
            cv_finalize_contour(ctx, ctx->contour);
        }
        ctx->contour = idx;
        ctx->edge = -1;
        ctx->contour_area = 0.f;
        ctx->contour_min.x = FLT_MAX;
        ctx->contour_min.y = FLT_MAX;
        ctx->contour_max.x = -FLT_MAX;
        ctx->contour_max.y = -FLT_MAX;
        break;
    case cv_type_2d_edge_linear:
    case cv_type_2d_edge_cubic:
    case cv_type_2d_edge_conic:
        if (ctx->edge < idx) {
            cv_node_set_next(cv_node_array_item(ctx, ctx->edge), idx);
        }
        ctx->edge = idx;

        /* accumulate contour area, min_x, min_y, max_x, max_y */
        pc = cv_point_count(type);
        for(size_t i = 0; i < pc; i++) {
            p = cv_point_array_item(ctx, offset + i);
            ctx->contour_min.x = cv_min(ctx->contour_min.x, p->v.x);
            ctx->contour_min.y = cv_min(ctx->contour_min.y, p->v.y);
            ctx->contour_max.x = cv_max(ctx->contour_max.x, p->v.x);
            ctx->contour_max.y = cv_max(ctx->contour_max.y, p->v.y);
            if (i == 0) p1 = p;
            if (i == pc-1) p2 = p;
        }
        ctx->contour_area += (p1->v.x * p2->v.y - p2->v.x * p1->v.y) / 2.f;

        break;
    }
    return idx;
}

static void cv_finalize_contour(cv_manifold *ctx, int contour)
{
    cv_node *node = cv_node_array_item(ctx, contour);

    /* stash contour min and max in reserved point slots */
    cv_point *pminmax = cv_point_array_item(ctx, cv_node_offset(node));
    pminmax[0].v.x = ctx->contour_min.x;
    pminmax[0].v.y = ctx->contour_min.y;
    pminmax[1].v.x = ctx->contour_max.x;
    pminmax[1].v.y = ctx->contour_max.y;

    /* set winding on contour */
    if (ctx->contour_area < 0) cv_node_set_attr(node, cv_contour_cw);
    else if (ctx->contour_area > 0) cv_node_set_attr(node, cv_contour_ccw);

    /* accumulate shape min and max */
    ctx->shape_area += ctx->contour_area;
    ctx->shape_min.x = cv_min(ctx->shape_min.x, ctx->contour_min.x);
    ctx->shape_min.y = cv_min(ctx->shape_min.y, ctx->contour_min.y);
    ctx->shape_max.x = cv_max(ctx->shape_max.x, ctx->contour_max.x);
    ctx->shape_max.y = cv_max(ctx->shape_max.y, ctx->contour_max.y);
}

static void cv_finalize_shape(cv_manifold *ctx, int shape)
{
    /* stash shape min and max in reserved point slots */
    cv_node *node = cv_node_array_item(ctx, shape);
    cv_point *pminmax = cv_point_array_item(ctx, cv_node_offset(node));
    pminmax[0].v.x = ctx->shape_min.x;
    pminmax[0].v.y = ctx->shape_min.y;
    pminmax[1].v.x = ctx->shape_max.x;
    pminmax[1].v.y = ctx->shape_max.y;
}

static uint cv_new_glyph(cv_manifold *ctx, int codepoint, int shape, float size,
    float width, float height, float offset_x, float offset_y,
    float advance_x, float advance_y)
{
    uint idx = array_buffer_count(&ctx->glyphs);
    cv_glyph g = {
        codepoint, shape, size, width, height,
        offset_x, offset_y, advance_x, advance_y
    };
    array_buffer_add(&ctx->glyphs, &g);
    array_buffer_resize(&ctx->codes, codepoint + 1);
    cv_code_array_item(ctx, codepoint)->glyph = idx;
    return idx;
}

static uint cv_lookup_glyph(cv_manifold *ctx, uint codepoint)
{
    uint ncodes = array_buffer_count(&ctx->codes);
    if (codepoint < ncodes) {
        return cv_code_array_item(ctx, codepoint)->glyph;
    }
    return 0;
}

/*
 * manifold buffer traversal
 */

static void cv_traverse_nodes(cv_manifold *ctx, uint idx, uint end, uint depth, void *userdata,
    int (*node_cb)(cv_manifold *ctx, uint idx, uint end, uint depth, void *userdata))
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx, idx);
        uint next = cv_node_next(node);
        uint type = cv_node_type(node);
        uint down_idx = idx + 1, down_end = next ? next : end;
        if (node_cb(ctx, idx, down_end, depth, userdata) && down_idx < down_end)
            switch (type) {
            case cv_type_2d_shape:
            case cv_type_2d_contour:
                cv_traverse_nodes(ctx, down_idx, down_end, depth + 1,
                                  userdata, node_cb);
                break;
            };
        idx = next ? next : end;
    }
}

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

static int cv_print_node(cv_manifold *ctx, uint idx, uint end, uint depth, void *userdata)
{
    cv_node *node = cv_node_array_item(ctx, idx);
    uint type = cv_node_type(node);
    uint point_offset = cv_node_offset(node);
    uint point_count = cv_point_count(type);
    char name[32];

    cv_info("graph: [%-6u] ", idx);
    for (uint i = 0; i < depth; i++) cv_info("    ");
    switch (type) {
    case cv_type_2d_contour:
        snprintf(name, sizeof(name), "%s (%s)",
            cv_node_type_name(node), cv_contour_attr_name(node));
        break;
    default:
        snprintf(name, sizeof(name), "%s",
            cv_node_type_name(node));
        break;
    }
    cv_info("+ %-17s", name);
    for (uint i = 0; i < 3-depth; i++) cv_info("    ");

    if (point_count > 0) {
        cv_info("[%-6u]", point_offset);
        for (uint k = 0; k < point_count; k++) {
            cv_point *p = cv_point_array_item(ctx, point_offset + k);
            cv_info(" (%7.3f,%7.3f)", p->v.x, p->v.y);
        }
    }
    cv_info("\n");

    return 1;
}

static void cv_dump_header()
{
    cv_info("graph: %-8s %-30s %-8s\n",
        "id", "name", "offset");
    cv_info("graph: %-8s %-30s %-8s\n",
        "--------", "------------------------------", "--------");
}

static void cv_dump_graph(cv_manifold *ctx)
{
    cv_dump_header();
    uint end = array_buffer_count(&ctx->nodes);
    cv_traverse_nodes(ctx, 0, end, 0, 0, cv_print_node);
}

static void cv_dump_node(cv_manifold *ctx, uint idx)
{
    cv_node *node = cv_node_array_item(ctx, idx);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&ctx->nodes);
    cv_traverse_nodes(ctx, idx, end, 0, 0, cv_print_node);
}

static void cv_dump_stats(cv_manifold *ctx)
{
    cv_info("stats: %8s %8s %8s %8s\n",
        "", "glyphs",   "nodes",    "points");
    cv_info("stats: %8s %8s %8s %8s\n",
        "", "--------",   "--------", "--------");
    cv_info("stats: %8s %8zu %8zu %8zu\n",
        "count",
        array_buffer_count(&ctx->glyphs),
        array_buffer_count(&ctx->nodes),
        array_buffer_count(&ctx->points));
    cv_info("stats: %8s %8zu %8zu %8zu\n",
        "size",
        array_buffer_size(&ctx->glyphs),
        array_buffer_size(&ctx->nodes),
        array_buffer_size(&ctx->points));
}

/*
 * freetype outline callbacks
 */

static uint ft_point(ft_vec *v, void *u)
{
    cv_manifold *ctx = (cv_manifold*)u;
    return cv_new_point(ctx, v->x/64.0f, v->y/64.0f);
}

static int ft_move_to(ft_vec *p, void *u)
{
    cv_manifold *ctx = (cv_manifold*)u;
    uint p1 = cv_new_points(ctx, 2); // reserve for minmax
    ctx->point = ft_point(p, u);
    cv_new_node(ctx, cv_type_2d_contour, (uint)p1);
    return 0;
}

static int ft_line_to(ft_vec *p, void *u)
{
    cv_manifold *ctx = (cv_manifold*)u;
    uint p1 = ft_point(p, u);
    cv_new_node(ctx, cv_type_2d_edge_linear, (uint)ctx->point);
    ctx->point = p1;
    return 0;
}

static int ft_conic_to(ft_vec *c, ft_vec *p, void *u)
{
    cv_manifold *ctx = (cv_manifold*)u;
    uint p1 = ft_point(c, u), p2 = ft_point(p, u);
    cv_new_node(ctx, cv_type_2d_edge_conic, (uint)ctx->point);
    ctx->point = p2;
    return 0;
}

static int ft_cubic_to(ft_vec *c1, ft_vec *c2, ft_vec *p, void *u)
{
    cv_manifold *ctx = (cv_manifold*)u;
    uint p1 = ft_point(c1, u), p2 = ft_point(c2, u), p3 = ft_point(p, u);
    cv_new_node(ctx, cv_type_2d_edge_cubic, (uint)ctx->point);
    ctx->point = p3;
    return 0;
}

static vec2f ft_glyph_size(FT_Glyph_Metrics *m)
{
    vec2f v = {{ (float)m->width/64.0f, (float)m->height/64.0f }};
    return v;
}

static vec2f ft_glyph_bearing(FT_Glyph_Metrics *m)
{
    vec2f v = {{ (float)m->horiBearingX/64.0f, (float)m->horiBearingY/64.0f }};
    return v;
}

/*
 * freetype glyph loading
 */

static FT_Library cv_init_ftlib()
{
    FT_Error fterr;
    FT_Library ftlib;

    if ((fterr = FT_Init_FreeType(&ftlib)))
        cv_panic("error: FT_Init_FreeType failed: fterr=%d\n", fterr);

    return ftlib;
}

static void cv_done_ftlib(FT_Library ftlib)
{
    FT_Done_Library(ftlib);
}

static FT_Face cv_load_ftface(FT_Library ftlib, const char* fontpath)
{
    FT_Error fterr;
    FT_Face ftface;

    if ((fterr = FT_New_Face(ftlib, fontpath, 0, &ftface)))
        cv_panic("error: FT_New_Face failed: fterr=%d, path=%s\n",
            fterr, fontpath);

    FT_Select_Charmap(ftface, FT_ENCODING_UNICODE);

    return ftface;
}

static void cv_done_ftface(FT_Face ftface)
{
    FT_Done_Face(ftface);
}

static uint cv_load_ftglyph(cv_manifold *ctx, FT_Face ftface,
    float font_size, int dpi, int codepoint)
{
    FT_Outline_Funcs ftfuncs = { ft_move_to, ft_line_to, ft_conic_to, ft_cubic_to, 0, 0 };
    FT_Glyph_Metrics *m = &ftface->glyph->metrics;
    FT_Error fterr;

    int size = font_size * 64.0f;
    int glyph_index = FT_Get_Char_Index(ftface, codepoint);

    if ((fterr = FT_Set_Char_Size(ftface, size, size, dpi, dpi))) {
        cv_error("error: FT_Set_Char_Size failed: fterr=%d\n", fterr);
        return -1;
    }

    if ((fterr = FT_Load_Glyph(ftface, glyph_index, 0))) {
        cv_error("error: FT_Load_Glyph failed: fterr=%d\n", fterr);
        return -1;
    }

    FT_Matrix  matrix = { 1 * 0x10000L, 0, 0, -1 * 0x10000L };
    FT_Outline_Transform(&ftface->glyph->outline, &matrix);

    uint p1 = cv_new_points(ctx, 2); // reserve for minmax
    uint shape = cv_new_node(ctx, cv_type_2d_shape, (uint)p1);
    uint glyph = cv_new_glyph(ctx, codepoint, shape,
        font_size, (float)m->width/64.0f, (float)m->height/64.0f,
        (float)m->horiBearingX/64.0f, (float)m->horiBearingY/64.0f,
        ftface->glyph->advance.x/64.0f, ftface->glyph->advance.y/64.0f);

    if ((fterr = FT_Outline_Decompose(&ftface->glyph->outline, &ftfuncs, ctx)))
        cv_panic("error: FT_Outline_Decompose failed: fterr=%d\n", fterr);

    if (ctx->contour < array_buffer_count(&ctx->nodes)) {
        cv_finalize_contour(ctx, ctx->contour);
    }
    if (ctx->shape < array_buffer_count(&ctx->nodes)) {
        cv_finalize_shape(ctx, ctx->shape);
    }

    return glyph;
}

static void cv_load_ftglyph_all(cv_manifold *ctx, FT_Face ftface,
    float font_size, int dpi)
{
    FT_UInt idx;
    FT_ULong charcode = FT_Get_First_Char(ftface, &idx);
    while (idx != 0) {
        cv_load_ftglyph(ctx, ftface, font_size, dpi, charcode);
        charcode = FT_Get_Next_Char(ftface, charcode, &idx);
    }
}

static char** cv_split_buffer_items(const char* buffer,
    size_t buffer_size, size_t* num_items_out, char* sep_chars)
{
    size_t n = 0, start = 0;
    for (size_t i = 0; i < buffer_size; i++) {
        int is_sep = strchr(sep_chars, buffer[i]) != NULL;
        if (!is_sep) continue;
        size_t length = i - start;
        if (length > 0) n++;
        start = i + 1;
    }
    n += strchr(sep_chars, buffer[buffer_size-1]) == NULL;
    start = 0;
    char** items = (char**)calloc(sizeof(char*) * (n + 1) + buffer_size, 1);
    char* item_buffer = (char*)items + sizeof(char*) * (n + 1);
    size_t item = 0;
    for (size_t i = 0; i < buffer_size; i++) {
        int is_sep = strchr(sep_chars, buffer[i]) != NULL;
        if (!is_sep) continue;
        size_t length = i - start;
        if (length > 0) {
            memcpy(item_buffer, &buffer[start], length);
            item_buffer[length] = '\0';
            items[item++] = item_buffer;
            item_buffer += length + 1;
        }
        start = i + 1;
    }
    if (start != buffer_size) {
        size_t length = buffer_size - start;
        if (length > 0) {
            memcpy(item_buffer, &buffer[start], length);
            item_buffer[length] = '\0';
            items[item++] = item_buffer;
            item_buffer += length + 1;
        }
    }
    *num_items_out = n;
    return items;
}

static uint cv_load_glyph_text_buffer(cv_manifold *ctx, buffer buf, int codepoint)
{
    size_t num_lines = 0;
    char **lines = cv_split_buffer_items(buf.data, buf.length, &num_lines, "\n");
    for (size_t i = 0; i < num_lines; i++) {
        size_t num_tokens = 0;
        char **tokens = cv_split_buffer_items(lines[i], strlen(lines[i]), &num_tokens, " ");
        for (size_t j = 0; j < num_tokens; j++) {
            if (strcmp("begin_shape", tokens[j]) == 0) {
                uint p1 = cv_new_points(ctx, 2); // reserve for minmax
                ctx->shape = cv_new_node(ctx, cv_type_2d_shape, (uint)p1);
            } else if (strcmp("end_shape", tokens[j]) == 0) {
                cv_finalize_shape(ctx, ctx->shape);
            } else if (strcmp("begin_contour", tokens[j]) == 0) {
                uint p1 = cv_new_points(ctx, 2); // reserve for minmax
                ctx->contour = cv_new_node(ctx, cv_type_2d_contour, (uint)p1);
            } else if (strcmp("end_contour", tokens[j]) == 0) {
                cv_finalize_contour(ctx, ctx->contour);
            } else if (strcmp("move_to", tokens[j]) == 0) {
                float x = (float)atof(tokens[1]), y = (float)atof(tokens[2]);
                ctx->point = cv_new_point(ctx, x, y);
            } else if (strcmp("line_to", tokens[j]) == 0) {
                float x = (float)atof(tokens[1]), y = (float)atof(tokens[2]);
                uint p1 = cv_new_point(ctx, x, y);
                cv_new_node(ctx, cv_type_2d_edge_linear, (uint)ctx->point);
                ctx->point = p1;
            }
        }
        free(tokens);
    }
    free(lines);
    uint glyph = cv_new_glyph(ctx, codepoint, ctx->shape, 0, 0, 0, 0, 0, 0, 0);
    return glyph;
}

static uint cv_load_glyph_text_file(cv_manifold *ctx, const char* filename,
    int codepoint)
{
    buffer buf = load_file(filename);
    int glyph = cv_load_glyph_text_buffer(ctx, buf, codepoint);
    free(buf.data);
    return glyph;
}

static uint cv_dump_metrics(cv_manifold *ctx, uint glyph)
{
    cv_glyph *g = cv_glyph_array_item(ctx, glyph);
    cv_info("metrics: %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n",
        "glyph", "unicode", "shape", "size", "width",
        "height", "off_x", "off_x", "adv_x", "adv_y");
    cv_info("metrics: %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n",
        "--------", "--------", "--------", "--------", "--------",
        "--------", "--------", "--------", "--------", "--------");
    cv_info("metrics: %8u %8u %8u %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n",
        glyph, g->codepoint, g->shape, g->size, g->width, g->height,
        g->offset_x, g->offset_y, g->advance_x, g->advance_y);
}

/*
 * hull contour rotation
 */

static int cv_hull_rotate_contour(cv_manifold *ctx, uint idx, uint end, uint offset)
{
    cv_node *node = cv_node_array_item(ctx, idx);
    uint next = cv_node_next(node);

    /* rotate edges */
    uint edge_idx = idx + 1, edge_end = next ? next : end;
    uint n = edge_idx < edge_end ? edge_end - edge_idx : 0;
    cv_node *edges = (cv_node*)alloca(sizeof(cv_node) * n);
    for (uint i = 0; i < n; i++) {
        cv_node *edge_a = cv_node_array_item(ctx, idx + 1 + i);
        cv_node *edge_b = cv_node_array_item(ctx, idx + 1 + (i + offset) % n);
        uint type_next = (cv_node_type(edge_b) << 28) | cv_node_next(edge_a);
        uint attr_offset = (cv_node_attr(edge_b) << 28) | cv_node_offset(edge_b);
        edges[i] = (cv_node) { type_next, attr_offset };
    }
    for (uint i = 0; i < n; i++) {
        cv_node *edge = cv_node_array_item(ctx, idx + 1 + i);
        *edge = edges[i];
    }
}

static int cv_hull_rotate_contours(cv_manifold *ctx, uint idx, uint end, uint offset)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx, idx);
        cv_trace("cv_hull_rotate_contours: contour=%s_%u offset=%u\n",
            cv_node_type_name(node), idx, offset);
        uint next = cv_node_next(node);
        cv_hull_rotate_contour(ctx, idx, end, offset);
        idx = next ? next : end;
    }
    return 0;
}

static int cv_hull_rotate_shapes(cv_manifold *ctx, uint idx, uint end, uint offset)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx, idx);
        cv_trace("cv_hull_rotate_shapes: shape=%s_%u offset=%u\n",
            cv_node_type_name(node), idx, offset);
        uint next = cv_node_next(node);
        uint contour_idx = idx + 1, contour_end = next ? next : end;
        if (contour_idx < contour_end) {
            cv_hull_rotate_contours(ctx, contour_idx, contour_end, offset);
        }
        idx = next ? next : end;
    }
    return 0;
}

static uint cv_hull_rotate(cv_manifold *ctx, uint idx, uint offset)
{
    cv_node *node = cv_node_array_item(ctx, idx);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&ctx->nodes);
    cv_hull_rotate_shapes(ctx, idx, end, offset);
}

/*
 * hull conversion
 */

enum cv_hull_transform_opt
{
    cv_hull_transform_forward = 0,
    cv_hull_transform_reverse = 1,
};

typedef struct cv_transform cv_transform;
struct cv_transform
{
    cv_manifold *src;
    cv_manifold *dst;
    array_buffer visited;
};

static int cv_extract_sign(float v)
{
    return (v < 0.f) ? -1 : (v > 0.f) ? 1 : 0;
}

static vec2f cv_point_get(cv_manifold *ctx, uint point)
{
    return cv_point_array_item(ctx, point)->v;
}

static void cv_hull_point_debug_header()
{
    cv_debug("hull: %-8s %-6s %17s %17s %17s %17s %7s %7s %7s\n",
             "id", "split", "a(curr)",
             "b(next)", "c(close)", "d(first)",
             "ab", "bc", "cd");
    cv_debug("hull: %-8s %-6s %-17s %-17s %-17s %-17s %-7s %-7s %-7s\n",
             "--------", "------", "-----------------",
             "-----------------", "-----------------", "-----------------",
             "-------", "-------", "-------");
}

static int cv_hull_point(uint *pl, int n, int i) { return pl[i % n]; }

static void cv_hull_point_debug_row(cv_manifold *mb, uint *pl, int n, int i,
    int split_idx, vec2f a, vec2f b, vec2f c, vec2f d, float ab, float bc, float cd)
{
    char str[16] = {};
    if (split_idx != -1) {
        snprintf(str, sizeof(str), "%-6d", cv_hull_point(pl, n, split_idx));
    }
    cv_debug("hull: [%-6u] %-6s (%7.3f,%7.3f) (%7.3f,%7.3f) "
        "(%7.3f,%7.3f) (%7.3f,%7.3f) %7.1f %7.1f %7.1f\n",
        cv_hull_point(pl, n, i), str,
        a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, ab, bc, cd);
}

/*
 * béziergon skip concave edges
 *
 * algorithm to split skip over concave sections.
 * *
 * compares the polarity of the cross product of the last edge
 * with the current edge to detect changes in winding order.
 */

static int cv_hull_skip_contour(cv_manifold* mb, uint *pl, int n,
    int s, int e, int dir, int w)
{
    if (cv_ll <= cv_ll_debug) {
        cv_hull_point_debug_header();
    }

    /* state variables */
    int split_idx = -1;

    int j0 = (s + n) % n;
    int j1 = (s + n + dir) % n;

    cv_trace("hull: first edge %d:%d dir=%d winding=%d\n",
        cv_hull_point(pl, n, j0), cv_hull_point(pl, n, j1), dir, w);

    /* find convex hull split point where winding order diverges */
    for (int i = s; i != e && split_idx == -1; i += dir)
    {
        int i1 = (i + n) % n;
        int i2 = (i + n + dir) % n;
        int i3 = (i + n + dir + dir) % n;

        vec2f v1 = cv_point_get(mb, pl[i1]);
        vec2f v2 = cv_point_get(mb, pl[i2]);
        vec2f v3 = cv_point_get(mb, pl[i3]);

        vec2f a = (vec2f) { v2.x - v1.x, v2.y - v1.y }; /* curr */
        vec2f b = (vec2f) { v3.x - v2.x, v3.y - v2.y }; /* next */

        float ab = vec2f_cross(a, b); /* cross of curr->next */

        int ab_w = cv_extract_sign(ab); /* winding of curr->next */

        /* winding of curr->next edge non-convex */
        if (ab_w && ab_w != w) split_idx = i;
    }

    return split_idx;
}

/*
 * béziergon convex interior hulls
 *
 * algorithm to split beziergon contours into convex interior hulls.
 *
 * start at some edge and walk around the contour testing the cross product
 * of each edge against a closure of subsequent edge vectors to choose a
 * convex split point if the winding order changes, then test if subsequent
 * edges intersect the hull, shrinking it if necessary.
 *
 * there are two phases to the algorithm:
 *
 * - phase one - find convex hull split point where winding order diverges.
 * - phase two - shrink hull so it is not intersected by subsequent edges.
 *
 * during phase one, the polarity of the cross product of the last edge with
 * the current edge, the current edge with next edge, the next edge with an
 * inserted closing edge and an inserted closing edge with the first edge are
 * all checked to detect changes in winding order, thus changes in convexity.
 * if any of these change in polarity the next edge will start a new hull.
 *
 * during phase two, each subsequent point along the contour is tested
 * against the edges of the current hull to see if they are inside the hull.
 * if a subsequent point on the contour is inside the hull, the hull is
 * shrunk by backing up to the last edge that no longer contains that point.
 *
 * work in progress. code presently only finds the first convex section.
 * and does not yet take into consideration the control points.
 */

static int opt_tracing = 0;
static int opt_epsilon = 1;

static int cv_hull_trace_contour(cv_manifold* mb, uint *pl, int n,
    int s, int i, int e, int dir, int w)
{
    if (cv_ll <= cv_ll_debug) {
        cv_hull_point_debug_header();
    }

    /* state variables */
    int split_idx = -1;

    int j0 = (s + n) % n;
    int j1 = (s + n + dir) % n;

    vec2f p0 = cv_point_get(mb, pl[j0]);
    vec2f p1 = cv_point_get(mb, pl[j1]);

    vec2f d = (vec2f) { p1.x - p0.x, p1.y - p0.y };

    /* find convex hull split point where winding order diverges */
    for (; i != e && split_idx == -1; i += dir)
    {
        int i1 = (i + n) % n;
        int i2 = (i + n + dir) % n;
        int i3 = (i + n + dir + dir) % n;

        vec2f v1 = cv_point_get(mb, pl[i1]);
        vec2f v2 = cv_point_get(mb, pl[i2]);
        vec2f v3 = cv_point_get(mb, pl[i3]);

        vec2f a = (vec2f) { v2.x - v1.x, v2.y - v1.y }; /* curr */
        vec2f b = (vec2f) { v3.x - v2.x, v3.y - v2.y }; /* next */
        vec2f c = (vec2f) { p0.x - v3.x, p0.y - v3.y }; /* close */

        float ab = vec2f_cross(a, b); /* cross of curr->next */
        float bc = vec2f_cross(b, c); /* cross of next->close */
        float cd = vec2f_cross(c, d); /* cross of close->first */

        int ab_w = cv_extract_sign(ab); /* winding of curr->next */
        int bc_w = cv_extract_sign(bc); /* winding of next->close */
        int cd_w = cv_extract_sign(cd); /* winding of close->first */

        if (split_idx == -1) {
            /* winding of prev->curr edge non-convex */
            if (ab_w && ab_w != w) {
                split_idx = i + dir;
                cv_trace("hull: edge %u prev->curr non convex\n",
                    cv_hull_point(pl, n, split_idx));
            }
            /* winding of curr->close edge non-convex */
            else if (bc_w && bc_w != w) {
                split_idx = i + dir;
                cv_trace("hull: edge %u curr->close non convex\n",
                    cv_hull_point(pl, n, split_idx));
            }
            /* winding of closing->first edge non-convex */
            else if (cd_w && cd_w != w) {
                split_idx = i + dir;
                cv_trace("hull: edge %u closing->first non convex\n",
                    cv_hull_point(pl, n, split_idx));
            }
        }

        if (cv_ll <= cv_ll_debug) {
            cv_hull_point_debug_row(mb, pl, n, i, split_idx,
                a, b, c, d, ab, bc, cd);
        }
    }

    /* shrink hull so it is not intersected by subsequent edges */
    for (; i != e && split_idx != -1; i += dir)
    {
        int i1 = (i + n) % n;
        int i2 = (i + n + dir) % n;
        int i3 = (i + n + dir + dir) % n;

        vec2f v1 = cv_point_get(mb, pl[i1]);
        vec2f v2 = cv_point_get(mb, pl[i2]);
        vec2f v3 = cv_point_get(mb, pl[i3]);

        vec2f a = (vec2f) { v2.x - v1.x, v2.y - v1.y }; /* curr */
        vec2f b = (vec2f) { v3.x - v2.x, v3.y - v2.y }; /* next */
        vec2f c = (vec2f) { p0.x - v3.x, p0.y - v3.y }; /* close */

        float ab = vec2f_cross(a, b); /* cross of curr->next */
        float bc = vec2f_cross(b, c); /* cross of next->close */
        float cd = vec2f_cross(c, d); /* cross of close->first */

        int ab_w = cv_extract_sign(ab); /* winding of curr->next */
        int bc_w = cv_extract_sign(bc); /* winding of next->close */
        int cd_w = cv_extract_sign(cd); /* winding of close->first */

        /* walk backwards from the convex split point reducing the
         * size of the hull until the current point is not inside it */
        uint inside = 1, new_split_idx = split_idx;
        do {
            split_idx = new_split_idx;
            int mmin = cv_min(s, split_idx + dir);
            int mmax = cv_max(s, split_idx + dir);
            int b = mmin, m = mmax - mmin;
            if (opt_tracing) {
                cv_trace("hull: contract %d:%d dir %d\n",
                    pl[s%n], pl[split_idx%n], dir);
            }
            for (uint j = s;
                inside && ((dir == 1 && j < split_idx + dir) ||
                          (dir == -1 && j > split_idx + dir));
                j += dir)
            {
                /* end point of the last edge is stored in the next
                 * edge so we must wrap split_idx + dir to start.
                 *
                 * modulus m is the relative range of the hull
                 * the zero index wraps back to the start index */
                int j1 = (j - b + m) % m;
                int j2 = (j - b + m + dir) % m;

                int k1 = (j1 == 0 ? s : b + j1) % n;
                int k2 = (j2 == 0 ? s : b + j2) % n;

                vec2f p1 = cv_point_get(mb, pl[k1]);
                vec2f p2 = cv_point_get(mb, pl[k2]);

                float d = vec2f_line_dist(p1, p2, v1);
                float b = cv_extract_sign(d);

                if (opt_epsilon) {
                    if ((pl[i1] == pl[k1] || pl[i1] == pl[k2]) && b == 0.f) {
                        inside = 0;
                    } else {
                        switch (w) {
                        case -1: inside = b>-1e-9; break;
                        case 1: inside = b<1e-9; break;
                        }
                    }
                } else {
                    switch (w) {
                    case -1: inside = b>0; break;
                    case 1: inside = b<0; break;
                    }
                }
                if (opt_tracing) {
                    cv_trace("hull: contract %d inside %d:%d = %d (%7.3f)\n",
                        pl[i1], pl[k1], pl[k2], inside, b);
                }
            }
            if (inside) {
                cv_trace("hull: contour edge %u inside %u:%u\n",
                    cv_hull_point(pl, n, i),
                    cv_hull_point(pl, n, 0),
                    cv_hull_point(pl, n, split_idx));
            }
            new_split_idx = split_idx - dir;
        } while (inside && new_split_idx != s);

        if (cv_ll <= cv_ll_debug) {
            cv_hull_point_debug_row(mb, pl, n, i, split_idx,
                a, b, c, d, ab, bc, cd);
        }
    }

    return split_idx;
}

typedef struct cv_hull_range cv_hull_range;
struct cv_hull_range { int n, s, e; };

static cv_hull_range cv_hull_make(int n, int r1, int r2)
{
    cv_hull_range r = { n, r1, r2 }; return r;
}

static cv_hull_range cv_hull_invert(cv_hull_range p)
{
    cv_hull_range q = { p.n, p.e, p.s }; return q;
}

static int cv_hull_len(cv_hull_range r)
{
    return 1 + abs(r.s < r.e ? r.e - r.s : r.n - r.s + r.e);
}

static vec2f cv_edge_point(cv_manifold *ctx, uint idx, uint n)
{
    cv_node *edge_node = cv_node_array_item(ctx, idx);
    return cv_point_array_item(ctx, cv_node_offset(edge_node) + n)->v;
}

static uint cv_edge_point_count(cv_manifold *ctx, uint idx)
{
    cv_node *edge_node = cv_node_array_item(ctx, idx);
    return cv_point_count(cv_node_type(edge_node));
}

static void cv_interior_hull(cv_manifold *mb, uint *pl, uint *m, uint idx, uint end)
{
    vec2f v1, v2, v3, a, b, c, l = { 0.f };
    float xla, dla, la, xlb, dlb, lb, xbc, dbc, bc;

    cv_node *node = cv_node_array_item(mb, idx);
    uint next = cv_node_next(node);
    uint edge_idx = idx + 1, edge_end = next ? next : end;
    uint n = edge_idx < edge_end ? edge_end - edge_idx : 0;
    int w;

    switch(cv_node_attr(node)) {
    case cv_contour_cw: w = -1; break;
    case cv_contour_ccw: w = 1; break;
    default: w = 0; break;
    }

    uint npoints = 0, p;
    for (uint i = n-1; i < n+n; i++) {
        uint pc = cv_edge_point_count(mb, edge_idx + i%n);
        cv_node *edge_node = cv_node_array_item(mb, edge_idx + i%n);
        switch (pc) {
        case 2:
            p = cv_node_offset(cv_node_array_item(mb, edge_idx + i%n));
            v1 =  cv_point_get(mb, p + 0);
            v2 =  cv_point_get(mb, p + 1);
            a = (vec2f) { v2.x - v1.x, v2.y - v1.y };
            xla = vec2f_cross(l, a), dla = vec2f_dot(l, a), la = atan2f(xla, dla);

            if (i != n-1)
            {
                if (pl) {
                    pl[npoints] = p;
                }
                npoints += 1;
            }

            l = a;
            break;
        case 3:
            p = cv_node_offset(cv_node_array_item(mb, edge_idx + i%n));
            v1 =  cv_point_get(mb, p + 0);
            v2 =  cv_point_get(mb, p + 1);
            v3 =  cv_point_get(mb, p + 2);
            a = (vec2f) { v2.x - v1.x, v2.y - v1.y };
            b = (vec2f) { v3.x - v1.x, v3.y - v1.y };
            c = (vec2f) { v3.x - v2.x, v3.y - v2.y };
            xla = vec2f_cross(l, a), dla = vec2f_dot(l, a), la = atan2f(xla, dla);
            xlb = vec2f_cross(l, b), dlb = vec2f_dot(l, b), lb = atan2f(xlb, dlb);
            xbc = vec2f_cross(l, b), dbc = vec2f_dot(l, b), bc = atan2f(xlb, dlb);

            if (i != n-1)
            {
                if (la > lb) {
                    if (pl) {
                        pl[npoints + 0] = p + 0;
                        pl[npoints + 1] = p + 1;
                    }
                    npoints += 2;
                    l = c;
                } else {
                    if (pl) {
                        pl[npoints] = p;
                    }
                    npoints += 1;
                    l = b;
                }
            } else {
                if (bc > 0) {
                    l = c;
                } else {
                    l = b;
                }
            }

            break;
        }
    }
    if (m) *m = npoints;
}

static void cv_point_list_moduli(uint *pl, uint *mpl, uint *m,
    cv_hull_range p, int invert, int reverse)
{
    if (invert) {
        p = cv_hull_invert(p);
    }
    if (p.e < p.s) {
        p.e += p.n;
    }
    if (m) {
        *m = 1 + p.e - p.s;
    }
    if (!mpl) {
        return;
    }
    if (reverse) {
        for (int i=p.e, j=0; i >= p.s; i--, j++) {
            mpl[j] = pl[i%p.n];
        }
    } else {
        for (int i=p.s, j=0; i <= p.e; i++, j++) {
            mpl[j] = pl[i%p.n];
        }
    }
}

static uint cv_edge_list_count(cv_manifold *mb, uint idx, uint end)
{
    cv_node *node = cv_node_array_item(mb, idx);
    uint next = cv_node_next(node);
    uint edge_idx = idx + 1, edge_end = next ? next : end;
    return edge_idx < edge_end ? edge_end - edge_idx : 0;
}

static void cv_edge_list_enum(cv_manifold *mb, uint *il, uint idx, uint end)
{
    uint n = cv_edge_list_count(mb, idx, end);
    for (uint i = 0; i < n; i++) {
        il[i] = idx + 1 + i;
    }
}

static uint cv_point_list_count(cv_manifold *mb, uint idx, uint end)
{
    uint n;
    cv_interior_hull(mb, NULL, &n, idx, end);
    return n;
}

static void cv_point_list_enum(cv_manifold *mb, uint *pl, uint idx, uint end)
{
    cv_interior_hull(mb, pl, NULL, idx, end);
}

static cv_hull_range cv_hull_split_contour(cv_manifold *mb, uint *pl, uint n,
    uint idx, uint opts, uint w)
{
    int r1, r2, lr1, len;
    cv_hull_range r = { n, 0, n-1 }, nr = { n, 0 , n-1 };
    switch (opts) {
    case cv_hull_transform_forward:
        lr1 = r1 = 0;
        do {
            /* skip concave edges and degenerate hulls */
            lr1 = r1;
            r2 = cv_hull_skip_contour(mb,pl,n,r1,n+r1,1,-w);
            if (r2 != -1) {
                r1 = cv_hull_trace_contour(mb,pl,n,r2,r2,n+r2,1,w);
                if (r1 != -1) {
                    nr = cv_hull_make(n,r2,r1); len = cv_hull_len(nr);
                    if (len > 2) r = nr; else r1 = lr1 + n + 1;
                }
            }
            cv_trace("hull: fwd_loop idx=%d n=%d s=%d (%d) e=%d (%d) len=%d\n",
                idx, n, nr.s % n, pl[nr.s % n], nr.e % n, pl[nr.e % n], len);
        } while (r2 != -1 && r1 != -1 && len < 3 && r1 < n*n);
        if (r1 != -1) {
            /* grow hull in the opposite direction */
            r2 = cv_hull_trace_contour(mb,pl,n,n+r1,n+r2+1,r1,-1,-w);
            if (r2 != -1) {
                nr = cv_hull_make(n,r2,r1); len = cv_hull_len(nr);
                if (len > 2) r = nr;
            }
            cv_trace("hull: fwd_grow idx=%d n=%d s=%d (%d) e=%d (%d) len=%d\n",
                idx, n, nr.s % n, pl[nr.s % n], nr.e % n, pl[nr.e % n], len);
        }
        break;
    case cv_hull_transform_reverse:
        lr1 = r1 = 0;
        do {
            /* skip concave edges and degenerate hulls */
            lr1 = r1;
            r2 = cv_hull_skip_contour(mb,pl,n,n+r1,r1,-1,w);
            if (r2 != -1) {
                r1 = cv_hull_trace_contour(mb,pl,n,n+r2,n+r2,r2,-1,-w);
                if (r1 != -1) {
                    nr = cv_hull_make(n,r1,r2); len = cv_hull_len(nr);
                    if (len > 2) r = nr; else r1 = lr1 + n - 1;
                }
            }
            cv_trace("hull: rev_loop idx=%d n=%d s=%d (%d) e=%d (%d) len=%d\n",
                idx, n, nr.s % n, pl[nr.s % n], nr.e % n, pl[nr.e % n], len);
        } while (r2 != -1 && r1 != -1 && len < 3 && r1 < n*n);
        if (r1 != -1) {
            /* grow hull in the opposite direction */
            r2 = cv_hull_trace_contour(mb,pl,n,r1,n+r2-1,n+r1,1,w);
            if (r2 != -1) {
                nr = cv_hull_make(n,r1,r2); len = cv_hull_len(nr);
                if (len > 2) r = nr;
            }
            cv_trace("hull: rev_grow idx=%d n=%d s=%d (%d) e=%d (%d) len=%d\n",
                idx, n, nr.s % n, pl[nr.s % n], nr.e % n, pl[nr.e % n], len);
        }
        break;
    }

    return r;
}


static void cv_hull_log_point_list(const char *prefix, uint *pl, cv_hull_range p)
{
    if (p.e < p.s) {
        p.e += p.n;
    }
    cv_trace("%s", prefix);
    for (int i=p.s; i <= p.e; i++) {
        cv_trace(" %d", pl[i%p.n]);
    }
    cv_trace("\n");
}

static array_buffer cv_hull_split_contour_loop(cv_manifold *mb,
    uint idx, uint end, uint opts, uint maxstep)
{
    uint n = cv_point_list_count(mb,idx,end);
    uint *pl = (uint*)alloca(sizeof(uint) * n);
    cv_point_list_enum(mb,pl,idx,end);
    cv_node *node = cv_node_array_item(mb, idx);
    cv_hull_range hr, ir;
    uint *tpl, *mpl = (uint*)alloca(sizeof(vec2f)*n);
    int u, m, w, step = 0;

    array_buffer result;
    array_buffer_init(&result, sizeof(cv_hull_range), 16);

    switch(cv_node_attr(node)) {
    case cv_contour_cw: w = -1; break;
    case cv_contour_ccw: w = 1; break;
    default: w = 0; break;
    }

    switch(cv_node_attr(node)) {
    case cv_contour_cw:
    case cv_contour_ccw:
        while (step++ < maxstep) {
            hr = cv_hull_split_contour(mb, pl, n, idx, opts, w);
            ir = cv_hull_invert(hr);
            array_buffer_add(&result, &hr);
            cv_point_list_moduli(pl, mpl, &m, ir, 0, 0);
            if (cv_ll <= cv_ll_trace) {
                cv_hull_log_point_list("hull: convex    : ", pl, hr);
                cv_hull_log_point_list("hull: remaining : ", pl, ir);
                cv_trace("hull: step=%d n=%d len(convex)=%d len(remaining)=%d\n",
                    step, n, cv_hull_len(hr), cv_hull_len(ir));
            }
            tpl = pl; pl = mpl; mpl = tpl;
            u   = n;  n  = m;   m = u;
            switch (opts) {
            case cv_hull_transform_forward: opts = cv_hull_transform_reverse; break;
            case cv_hull_transform_reverse: opts = cv_hull_transform_forward; break;
            }
            if (cv_hull_len(ir) < 3) break;
        }
        break;
    }

    return result;
}

static int cv_hull_transform_contours(cv_transform *ctx, uint idx, uint end,
    uint opts, uint maxstep)
{
    cv_manifold *mb = ctx->src;
    while (idx < end) {
        uint n = cv_point_list_count(mb,idx,end);
        uint *pl = (uint*)alloca(sizeof(uint) * n);
        cv_point_list_enum(mb,pl,idx,end);
        cv_node *node = cv_node_array_item(mb, idx);
        cv_trace("cv_hull_transform_contours: %s_%u\n", cv_node_type_name(node), idx);
        cv_hull_range hr, ir;
        uint *tpl, *mpl = (uint*)alloca(sizeof(vec2f)*n);
        array_buffer result;
        int u, m;
        switch(cv_node_attr(node)) {
        case cv_contour_cw:
        case cv_contour_ccw:
            result = cv_hull_split_contour_loop(mb, idx, end,
                opts, maxstep);
            for (int i = 0; i < array_buffer_count(&result); i++) {
                hr = ((cv_hull_range*)array_buffer_data(&result))[i];
                ir = cv_hull_invert(hr);
                // todo - create dest tree nodes for edge loop moduli
                cv_point_list_moduli(pl, mpl, &m, ir, 0, 0);
                tpl = pl; pl = mpl; mpl = tpl;
                u   = n;  n  = m;   m = u;
            }
            array_buffer_destroy(&result);
            break;
        }
        uint next = cv_node_next(node);
        idx = next ? next : end;
    }
    return 0;
}

static int cv_hull_transform_shapes(cv_transform *ctx, uint idx, uint end,
    uint opts, uint maxstep)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx->src, idx);
        cv_trace("cv_hull_transform_shapes: %s_%u\n", cv_node_type_name(node), idx);
        uint next = cv_node_next(node);
        uint contour_idx = idx + 1, contour_end = next ? next : end;
        if (contour_idx < contour_end) {
            cv_hull_transform_contours(ctx, contour_idx, contour_end, opts, maxstep);
        }
        idx = next ? next : end;
    }
    return 0;
}

static uint cv_hull_transform(cv_manifold *src, cv_manifold *dst, uint idx,
    uint opts, uint maxstep)
{
    cv_transform ctx = { src, dst };
    array_buffer_init(&ctx.visited, sizeof(char), array_buffer_count(&src->nodes));
    cv_node *node = cv_node_array_item(src, idx);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&src->nodes);
    cv_hull_transform_shapes(&ctx, idx, end, opts, maxstep);
}
