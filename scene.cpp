/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#include "scene.h"

#include <spdlog/spdlog.h>

#include <CGAL/IO/PLY.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace PMP = CGAL::Polygon_mesh_processing;

// ─────────────────────────────────────────────────────────────────────────────
// Colormap par seuils absolus de bruit [dB(A)]
//
//   < 35 dB  → Violet       (inaudible / bruit de fond)
//  35-45 dB  → Bleu         (très faible)
//  45-55 dB  → Jaune        (acceptable)
//  55-75 dB  → Orange       (gênant / attention)
//  75-85 dB  → Rouge        (nuisance)
//   > 85 dB  → Rouge foncé  (dangereux)
// ─────────────────────────────────────────────────────────────────────────────

static RGB lerpRGB(RGB a, RGB b, double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return { static_cast<uint8_t>(a.r + (b.r - a.r) * t),
             static_cast<uint8_t>(a.g + (b.g - a.g) * t),
             static_cast<uint8_t>(a.b + (b.b - a.b) * t) };
}

RGB splToColor(double spl)
{
    // Couleurs de référence pour chaque seuil
    static constexpr RGB COL_VIOLET    = { 128,   0, 200 };  // < 35 dB
    static constexpr RGB COL_BLUE      = {   0,  80, 255 };  // 35 dB
    static constexpr RGB COL_YELLOW    = { 255, 230,   0 };  // 45 dB
    static constexpr RGB COL_ORANGE    = { 255, 140,   0 };  // 55 dB
    static constexpr RGB COL_RED       = { 255,   0,   0 };  // 75 dB
    static constexpr RGB COL_DARK_RED  = { 100,   0,   0 };  // 85 dB

    if (spl < 0.0)  return COL_VIOLET;   // Violet
    if (spl < 15.0)  return COL_BLUE;     // Bleu
    if (spl < 25.0)  return COL_YELLOW;   // Jaune
    if (spl < 35.0)  return COL_ORANGE;   // Orange
    if (spl < 45.0)  return COL_RED;      // Rouge
                     return COL_DARK_RED;  // Rouge foncé

}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

Scene::Scene(const std::string& ply_path)
{
    spdlog::info("Scene: loading '{}'", ply_path);
    _loadPLY(ply_path);
    _repairMesh();

    spdlog::info("Scene: {} vertices, {} faces after repair",
                 mesh_.number_of_vertices(),
                 mesh_.number_of_faces());

    rayTracer_ = std::make_unique<RayTracer>(mesh_);
    spdlog::info("Scene: OptiX BVH ready");
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────

Scene::~Scene()
{
    if (rayTracer_)
        rayTracer_->cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// Load PLY
// ─────────────────────────────────────────────────────────────────────────────

void Scene::_loadPLY(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Scene: cannot open '" + path + "'");

    if (!CGAL::IO::read_PLY(in, mesh_))
        throw std::runtime_error("Scene: failed to parse PLY '" + path + "'");

    if (mesh_.is_empty())
        throw std::runtime_error("Scene: mesh is empty after loading '" + path + "'");

    spdlog::debug("_loadPLY: raw mesh has {} V / {} F",
                  mesh_.number_of_vertices(),
                  mesh_.number_of_faces());
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh repair
// ─────────────────────────────────────────────────────────────────────────────

void Scene::_repairMesh()
{
    PMP::remove_isolated_vertices(mesh_);
    PMP::stitch_borders(mesh_);
    PMP::remove_degenerate_faces(mesh_);
    mesh_.collect_garbage();

    centroidsDirty_ = true;

    spdlog::debug("_repairMesh done: {} V / {} F",
                  mesh_.number_of_vertices(),
                  mesh_.number_of_faces());
}

// ─────────────────────────────────────────────────────────────────────────────
// Face centroids (cached)
// ─────────────────────────────────────────────────────────────────────────────

void Scene::_buildCentroids() const
{
    centroids_.clear();
    centroids_.reserve(mesh_.number_of_faces());

    for (auto f : mesh_.faces()) {
        std::vector<Point> pts;
        pts.reserve(3);
        for (auto v : CGAL::vertices_around_face(mesh_.halfedge(f), mesh_))
            pts.push_back(mesh_.point(v));

        if (pts.size() == 3) {
            centroids_.push_back(CGAL::centroid(pts[0], pts[1], pts[2]));
        } else if (!pts.empty()) {
            double cx = 0, cy = 0, cz = 0;
            for (auto& p : pts) { cx += p.x(); cy += p.y(); cz += p.z(); }
            double n = static_cast<double>(pts.size());
            centroids_.emplace_back(cx / n, cy / n, cz / n);
        }
    }

    centroidsDirty_ = false;
    spdlog::debug("_buildCentroids: {} centroids", centroids_.size());
}

const std::vector<Point>& Scene::faceCentroids() const
{
    if (centroidsDirty_)
        _buildCentroids();
    return centroids_;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray tracing
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> Scene::traceRays(const Point& source) const
{
    spdlog::debug("traceRays: source=({:.3f},{:.3f},{:.3f})",
                  source.x(), source.y(), source.z());

    std::vector<float> distances = rayTracer_->traceRay(source);

    if (distances.size() != mesh_.number_of_faces()) {
        spdlog::warn("traceRays: distance count ({}) != face count ({})",
                     distances.size(), mesh_.number_of_faces());
    }

    // ── Debug : compte les faces visibles ────────────────────────────────────
    const float occluded_threshold = std::numeric_limits<float>::max() * 0.5f;
    size_t visible = std::count_if(distances.begin(), distances.end(),
        [&](float d){ return d > 0.0f && d < occluded_threshold; });

    spdlog::debug("traceRays: {}/{} faces visible (threshold={:.2e})",
                  visible, distances.size(), occluded_threshold);

    // ── Dump des 10 premières distances pour vérification ────────────────────
    for (size_t i = 0; i < std::min<size_t>(10, distances.size()); ++i)
        spdlog::trace("  dist[{}] = {:.4f}", i, distances[i]);

    return distances;
}

// ─────────────────────────────────────────────────────────────────────────────
// Acoustic computation
// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> Scene::computeNoiseMap(const std::vector<float>& distances,
                                           const AcousticModel&      model,
                                           const Point&              source) const
{
    const size_t N = distances.size();
    std::vector<double> spl(N, -std::numeric_limits<double>::infinity());

    const auto& ap     = model.params();
    const double scale = ap.unit_scale;

    const double hs_m = source.z() * scale;
    const auto& cents = faceCentroids();

    double spl_max = -std::numeric_limits<double>::infinity();
    double spl_min =  std::numeric_limits<double>::infinity();
    size_t vis_direct    = 0;
    size_t vis_reflected = 0;

    for (size_t i = 0; i < N; ++i) {
        const float d_raw        = distances[i];
        const bool direct_visible = (d_raw > 0.0f);

        if (i >= cents.size()) continue;
        const Point& c = cents[i];

        // ── Géométrie réelle face i ─────────────────────────────────────
        double dx       = (c.x() - source.x()) * scale;
        double dy       = (c.y() - source.y()) * scale;
        double hr_m     = c.z() * scale;            // hauteur récepteur [m]
        double dz_m     = hs_m - hr_m;              // diff. verticale [m]
        double d_horiz_m = std::sqrt(dx*dx + dy*dy);

        // ── Construit un modèle acoustique avec les hauteurs correctes ──
        // On clone les params et on met à jour source_height / receiver_height
        AcousticParams ap_face = ap;
        ap_face.source_height   = hs_m;
        ap_face.receiver_height = hr_m;
        AcousticModel model_face(ap_face);

        // ── Rayon réfléchi ───────────────────────────────────────────────
        double spl_reflected = -std::numeric_limits<double>::infinity();

        if (ap.reflection_order >= 1 && hr_m >= 0.0 && hs_m >= 0.0) {
            spl_reflected = model_face.computeReflectedSPL(
                                d_horiz_m, hs_m, hr_m);
        }

        if (direct_visible) {
            // Distance 3D réelle source → centroïde (ou du ray tracer)
            // Utilise d_raw du ray tracer car c'est la vraie distance
            // de propagation (le rayon peut longer un obstacle)
            const double d_direct_m = std::max(
                static_cast<double>(d_raw) * scale, 1e-3);

            // direct_only=true : groundEffect désactivé,
            // réflexion traitée séparément ci-dessus
            double spl_direct = model_face.computeSPL(
                                    d_direct_m, /*visible=*/true,
                                    /*direct_only=*/true);

            // Combinaison énergétique
            if (std::isfinite(spl_reflected)) {
                double e_d = std::pow(10.0, spl_direct    / 10.0);
                double e_r = std::pow(10.0, spl_reflected / 10.0);
                spl[i] = 10.0 * std::log10(e_d + e_r);
            } else {
                spl[i] = spl_direct;
            }
            ++vis_direct;

        } else if (std::isfinite(spl_reflected)) {
            spl[i] = spl_reflected;
            ++vis_reflected;
        }

        if (std::isfinite(spl[i])) {
            spl_max = std::max(spl_max, spl[i]);
            spl_min = std::min(spl_min, spl[i]);
        }
    }

    spdlog::info("computeNoiseMap: {}/{} direct, {}/{} reflected-only  "
                 "SPL min={:.1f}  max={:.1f} dB(A)",
                 vis_direct, N, vis_reflected, N,
                 std::isfinite(spl_min) ? spl_min : 0.0,
                 std::isfinite(spl_max) ? spl_max : 0.0);

    return spl;
}

// ─────────────────────────────────────────────────────────────────────────────
// Property : distances
// ─────────────────────────────────────────────────────────────────────────────

void Scene::addDistances(const std::vector<float>& distances)
{
    auto [prop, created] =
        mesh_.add_property_map<SurfaceMesh::Face_index, float>(
            "f:distance", std::numeric_limits<float>::max());

    size_t i = 0;
    for (auto f : mesh_.faces()) {
        if (i < distances.size())
            prop[f] = distances[i++];
    }

    spdlog::debug("addDistances: property 'f:distance' {} (created={})",
                  created ? "added" : "updated", created);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property : SPL
// ─────────────────────────────────────────────────────────────────────────────

void Scene::addSPL(const std::vector<double>& spl)
{
    auto [prop, created] =
        mesh_.add_property_map<SurfaceMesh::Face_index, double>(
            "f:spl_dBA", -std::numeric_limits<double>::infinity());

    size_t i = 0;
    for (auto f : mesh_.faces()) {
        if (i < spl.size())
            prop[f] = spl[i++];
    }

    spdlog::debug("addSPL: property 'f:spl_dBA' {} (created={})",
                  created ? "added" : "updated", created);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property : colour (face + vertex)
// ─────────────────────────────────────────────────────────────────────────────

void Scene::addNoiseMapColor(const std::vector<double>& spl)
{
    // ── Statistiques SPL (pour les logs) ─────────────────────────────────────
    double spl_min =  std::numeric_limits<double>::infinity();
    double spl_max = -std::numeric_limits<double>::infinity();

    for (double v : spl) {
        if (std::isfinite(v)) {
            spl_min = std::min(spl_min, v);
            spl_max = std::max(spl_max, v);
        }
    }

    if (!std::isfinite(spl_min)) {
        spdlog::warn("addNoiseMapColor: no visible face, colour map is meaningless");
        spl_min = 0.0;
        spl_max = 1.0;
    }

    spdlog::debug("addNoiseMapColor: SPL range [{:.1f}, {:.1f}] dB(A)",
                  spl_min, spl_max);

    // ── Couleur par face (seuils absolus) ────────────────────────────────────
    auto [fcolor, fc_created] =
        mesh_.add_property_map<SurfaceMesh::Face_index,
                               CGAL::IO::Color>(
            "f:color", CGAL::IO::Color(30, 30, 30));   // gris foncé = occluded

    {
        size_t i = 0;
        for (auto f : mesh_.faces()) {
            if (i < spl.size() && std::isfinite(spl[i])) {
                RGB c = splToColor(spl[i]);
                fcolor[f] = CGAL::IO::Color(c.r, c.g, c.b);
            }
            ++i;
        }
    }

    // ── Couleur par sommet : moyenne énergétique des faces incidentes ─────────
    auto [vcolor, vc_created] =
        mesh_.add_property_map<SurfaceMesh::Vertex_index,
                               CGAL::IO::Color>(
            "v:color", CGAL::IO::Color(30, 30, 30));

    // Accumulateurs
    const size_t NV = mesh_.number_of_vertices();
    std::vector<double> vEnergy(NV, 0.0);
    std::vector<int>    vCount (NV, 0);

    {
        size_t i = 0;
        for (auto f : mesh_.faces()) {
            if (i < spl.size() && std::isfinite(spl[i])) {
                double energy = std::pow(10.0, spl[i] / 10.0);
                for (auto v :
                     CGAL::vertices_around_face(mesh_.halfedge(f), mesh_)) {
                    vEnergy[v.idx()] += energy;
                    vCount [v.idx()] += 1;
                }
            }
            ++i;
        }
    }

    for (auto v : mesh_.vertices()) {
        if (vCount[v.idx()] > 0) {
            double avgSPL = 10.0 * std::log10(
                vEnergy[v.idx()] / static_cast<double>(vCount[v.idx()]));
            RGB c = splToColor(avgSPL);
            vcolor[v] = CGAL::IO::Color(c.r, c.g, c.b);
        }
    }

    // Onde directe uniquement : les faces occultées ne reçoivent aucun son.
    // Réinitialise les sommets de toute face occultée en gris foncé pour éviter
    // que l'interpolation des couleurs par sommet ne colore ces faces.
    {
        size_t i = 0;
        for (auto f : mesh_.faces()) {
            if (i >= spl.size() || !std::isfinite(spl[i])) {
                for (auto v : CGAL::vertices_around_face(mesh_.halfedge(f), mesh_))
                    vcolor[v] = CGAL::IO::Color(30, 30, 30);
            }
            ++i;
        }
    }

    spdlog::debug("addNoiseMapColor: face colour {} / vertex colour {}",
                  fc_created ? "added" : "updated",
                  vc_created ? "added" : "updated");
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounding box check
// ─────────────────────────────────────────────────────────────────────────────

bool Scene::isInsideBBox(const Point& p, double margin,
                         Point& bbox_min, Point& bbox_max) const
{
    double xmin =  std::numeric_limits<double>::infinity();
    double ymin =  std::numeric_limits<double>::infinity();
    double zmin =  std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();

    for (auto v : mesh_.vertices()) {
        const Point& pt = mesh_.point(v);
        xmin = std::min(xmin, pt.x()); xmax = std::max(xmax, pt.x());
        ymin = std::min(ymin, pt.y()); ymax = std::max(ymax, pt.y());
        zmin = std::min(zmin, pt.z()); zmax = std::max(zmax, pt.z());
    }

    bbox_min = Point(xmin, ymin, zmin);
    bbox_max = Point(xmax, ymax, zmax);

    return p.x() >= xmin - margin && p.x() <= xmax + margin
        && p.y() >= ymin - margin && p.y() <= ymax + margin
        && p.z() >= zmin - margin && p.z() <= zmax + margin;
}

// ─────────────────────────────────────────────────────────────────────────────
// Write PLY
// ─────────────────────────────────────────────────────────────────────────────

void Scene::writeMeshToPLY(const std::string& out_path) const
{
    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error(
            "Scene::writeMeshToPLY: cannot open '" + out_path + "' for writing");

    bool ok = CGAL::IO::write_PLY(out, mesh_,
                                  CGAL::parameters::stream_precision(6));
    if (!ok)
        throw std::runtime_error(
            "Scene::writeMeshToPLY: CGAL PLY write failed for '" + out_path + "'");

    spdlog::info("writeMeshToPLY: '{}' written ({} V, {} F)",
                 out_path,
                 mesh_.number_of_vertices(),
                 mesh_.number_of_faces());
}
