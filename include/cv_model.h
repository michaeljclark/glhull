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

typedef const FT_Vector ft_vec;

struct cv_meta { uint num_points, num_nodes; };
struct cv_node { uint type_next, attr_offset; };
struct cv_point { vec2f v; };
struct cv_glyph { uint codepoint, shape; float size, width, height;
                  float offset_x, offset_y, advance_x, advance_y; };

/*
 * simple math
 */

#define cv_min(a,b) (((a)<(b))?(a):(b))
#define cv_max(a,b) (((a)>(b))?(a):(b))

static float vec2f_length(vec2f v)
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

static float vec2f_cross_z(vec2f a, vec2f b)
{
    return a.x*b.y - b.x*a.y;
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

    FT_Face ftface;
};

static void cv_manifold_init(cv_manifold *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->shape = ctx->contour = ctx->edge = ctx->point = -1;
    array_buffer_init(&ctx->nodes, sizeof(cv_node), 16);
    array_buffer_init(&ctx->points, sizeof(cv_point), 16);
    array_buffer_init(&ctx->glyphs, sizeof(cv_glyph), 16);
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
    return idx;
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
    cv_info("stats: %8s %8zu %8zu %8zu\n\n",
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

static void cv_load_face(cv_manifold *ctx, const char* fontpath)
{
    FT_Error fterr;
    FT_Library ftlib;

    if ((fterr = FT_Init_FreeType(&ftlib)))
        cv_panic("error: FT_Init_FreeType failed: fterr=%d\n", fterr);

    if ((fterr = FT_New_Face(ftlib, fontpath, 0, &ctx->ftface)))
        cv_panic("error: FT_New_Face failed: fterr=%d, path=%s\n",
            fterr, fontpath);

    FT_Select_Charmap(ctx->ftface, FT_ENCODING_UNICODE);
}

static uint cv_load_one_glyph(cv_manifold *ctx, float font_size, int dpi, int codepoint)
{
    FT_Outline_Funcs ftfuncs = { ft_move_to, ft_line_to, ft_conic_to, ft_cubic_to, 0, 0 };
    FT_Glyph_Metrics *m = &ctx->ftface->glyph->metrics;
    FT_Error fterr;

    int size = font_size * 64.0f;
    int glyph_index = FT_Get_Char_Index(ctx->ftface, codepoint);

    if ((fterr = FT_Set_Char_Size(ctx->ftface, size, size, dpi, dpi)))
        cv_panic("error: FT_Set_Char_Size failed: fterr=%d\n", fterr);

    if ((fterr = FT_Load_Glyph(ctx->ftface, glyph_index, 0)))
        cv_panic("error: FT_Load_Glyph failed: fterr=%d\n", fterr);

    FT_Matrix  matrix = { 1 * 0x10000L, 0, 0, -1 * 0x10000L };
    FT_Outline_Transform(&ctx->ftface->glyph->outline, &matrix);

    uint p1 = cv_new_points(ctx, 2); // reserve for minmax
    uint shape = cv_new_node(ctx, cv_type_2d_shape, (uint)p1);
    uint glyph = cv_new_glyph(ctx, codepoint, shape,
        font_size, (float)m->width/64.0f, (float)m->height/64.0f,
        (float)m->horiBearingX/64.0f, (float)m->horiBearingY/64.0f,
        ctx->ftface->glyph->advance.x/64.0f, ctx->ftface->glyph->advance.y/64.0f);

    if ((fterr = FT_Outline_Decompose(&ctx->ftface->glyph->outline, &ftfuncs, ctx)))
        cv_panic("error: FT_Outline_Decompose failed: fterr=%d\n", fterr);

    cv_finalize_contour(ctx, ctx->contour);
    cv_finalize_shape(ctx, ctx->shape);

    return glyph;
}

static void cv_load_all_glyphs(cv_manifold *ctx, float font_size, int dpi)
{
    FT_UInt idx;
    FT_ULong charcode = FT_Get_First_Char(ctx->ftface, &idx);
    while (idx != 0) {
        cv_load_one_glyph(ctx, font_size, dpi, charcode);
        charcode = FT_Get_Next_Char(ctx->ftface, charcode, &idx);
    }
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
    uint n_edges = edge_idx < edge_end ? edge_end - edge_idx : 0;
    cv_node *edges = (cv_node*)alloca(sizeof(cv_node) * n_edges);
    for (uint i = 0; i < n_edges; i++) {
        cv_node *edge_a = cv_node_array_item(ctx, idx + 1 + i);
        cv_node *edge_b = cv_node_array_item(ctx, idx + 1 + (i + offset) % n_edges);
        uint type_next = (cv_node_type(edge_b) << 28) | cv_node_next(edge_a);
        uint attr_offset = (cv_node_attr(edge_b) << 28) | cv_node_offset(edge_b);
        edges[i] = (cv_node) { type_next, attr_offset };
    }
    for (uint i = 0; i < n_edges; i++) {
        cv_node *edge = cv_node_array_item(ctx, idx + 1 + i);
        *edge = edges[i];
    }
}

static int cv_hull_rotate_contours(cv_manifold *ctx, uint idx, uint end, uint offset)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx, idx);
        cv_debug("cv_hull_rotate_contours: contour=%s_%u offset=%u\n",
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
        cv_debug("cv_hull_rotate_shapes: shape=%s_%u offset=%u\n",
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
    cv_hull_transform_auto = 1,
    cv_hull_transform_forward = 2,
    cv_hull_transform_reverse = 3,
};

typedef struct cv_transform cv_transform;
struct cv_transform
{
    cv_manifold *src;
    cv_manifold *dst;
    array_buffer visited;
};

static float cv_extract_sign(float v)
{
    return (v < 0.f) ? -1.f : (v > 0.f) ? 1.f : 0.f;
}

static vec2f cv_edge_point(cv_manifold *ctx, uint idx)
{
    cv_node *edge_node = cv_node_array_item(ctx, idx);
    return cv_point_array_item(ctx, cv_node_offset(edge_node))->v;
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

static int cv_hull_trace_contour(cv_transform *ctx, uint contour_idx,
    vec2f *edges, int n_edges, int start, int end, int dir, int w)
{
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
    int split_at = 0;
    int edge_idx = contour_idx + 1;
    vec2f p0 = edges[start];
    vec2f p1 = edges[start + dir];
    vec2f d = (vec2f) { p1.x - p0.x, p1.y - p0.y };
    int i = start;

    /* find convex hull split point where winding order diverges */
    for (; i != end && split_at == 0; i += dir)
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

        if (split_at == 0) {
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
            cv_node *edge_node = cv_node_array_item(ctx->src, edge_idx + i);
            cv_debug("hull: [%-6u] %-16s %-6s (%7.3f,%7.3f) (%7.3f,%7.3f) "
                "(%7.3f,%7.3f) (%7.3f,%7.3f) %7.1f %7.1f %7.1f\n",
                edge_idx + i, cv_node_type_name(edge_node), "",
                a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, ab, bc, cd);
        }

        ab_sl = ab_s;
    }

    /* shrink hull so it is not intersected by subsequent edges */
    for (; i != end && split_at != 0; i += dir)
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
            cv_node *edge_node = cv_node_array_item(ctx->src, edge_idx + i);
            cv_debug("hull: [%-6u] %-16s %-6s (%7.3f,%7.3f) (%7.3f,%7.3f) "
                "(%7.3f,%7.3f) (%7.3f,%7.3f) %7.1f %7.1f %7.1f\n",
                edge_idx + i, cv_node_type_name(edge_node), str,
                a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, ab, bc, cd);
        }
    }

    return split_at;
}

static int cv_hull_transform_contour(cv_transform *ctx, uint idx, uint end, uint opts)
{
    cv_node *node = cv_node_array_item(ctx->src, idx);
    uint next = cv_node_next(node);

    uint edge_idx = idx + 1, edge_end = next ? next : end;
    uint n_edges = edge_idx < edge_end ? edge_end - edge_idx : 0;
    vec2f *edges = (vec2f*)alloca(sizeof(vec2f) * n_edges);
    for (uint i = 0; i < n_edges; i++) {
        edges[i] = cv_edge_point(ctx->src, edge_idx + i);
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
        r = cv_hull_trace_contour(ctx, idx, edges, n_edges,         0, n_edges,  1, w);
        break;
    case cv_hull_transform_forward:
        r = cv_hull_trace_contour(ctx, idx, edges, n_edges,         0, n_edges,  1, w);
        break;
    case cv_hull_transform_reverse:
        r = cv_hull_trace_contour(ctx, idx, edges, n_edges, n_edges-1,      -1, -1, -w);
        break;
    }

    return r;
}

static int cv_hull_transform_contours(cv_transform *ctx, uint idx, uint end, uint opts)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx->src, idx);
        cv_debug("cv_hull_transform_contours: %s_%u\n", cv_node_type_name(node), idx);
        uint next = cv_node_next(node);
        cv_hull_transform_contour(ctx, idx, end, opts);
        idx = next ? next : end;
    }
    return 0;
}

static int cv_hull_transform_shapes(cv_transform *ctx, uint idx, uint end, uint opts)
{
    while (idx < end) {
        cv_node *node = cv_node_array_item(ctx->src, idx);
        cv_debug("cv_hull_transform_shapes: %s_%u\n", cv_node_type_name(node), idx);
        uint next = cv_node_next(node);
        uint contour_idx = idx + 1, contour_end = next ? next : end;
        if (contour_idx < contour_end) {
            cv_hull_transform_contours(ctx, contour_idx, contour_end, opts);
        }
        idx = next ? next : end;
    }
    return 0;
}

static uint cv_hull_transform(cv_manifold *src, cv_manifold *dst, uint idx, uint opts)
{
    cv_transform ctx = { src, dst };
    array_buffer_init(&ctx.visited, sizeof(char), array_buffer_count(&src->nodes));
    cv_node *node = cv_node_array_item(src, idx);
    uint end = cv_node_next(node) ? cv_node_next(node)
                                  : array_buffer_count(&src->nodes);
    cv_hull_transform_shapes(&ctx, idx, end, opts);
}