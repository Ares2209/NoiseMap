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
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace PMP = CGAL::Polygon_mesh_processing;

namespace {

double vectorNorm(const Vector& v)
{
    return std::sqrt(v.squared_length());
}

Vector normalizedOrZero(const Vector& v)
{
    const double norm = vectorNorm(v);
    if (norm <= 1e-12)
        return Vector(0.0, 0.0, 0.0);
    return v / norm;
}

double dotProduct(const Vector& a, const Vector& b)
{
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

Vector reflectAroundNormal(const Vector& incident_to_surface,
                           const Vector& unit_normal)
{
    return incident_to_surface
         - 2.0 * dotProduct(incident_to_surface, unit_normal) * unit_normal;
}

// ln(10)/10 — used to replace pow(10, x/10) with the faster exp(x * LN10_OVER_10)
constexpr double LN10_OVER_10 = 0.23025850929940457;

double energySumDb(double a_db, double b_db)
{
    if (!std::isfinite(a_db)) return b_db;
    if (!std::isfinite(b_db)) return a_db;

    const double e_a = std::exp(a_db * LN10_OVER_10);
    const double e_b = std::exp(b_db * LN10_OVER_10);
    return 10.0 * std::log10(e_a + e_b);
}

}  // namespace

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
    PMP::triangulate_faces(mesh_);
    mesh_.collect_garbage();

    centroidsDirty_ = true;
    normalsDirty_   = true;

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
        // All faces are triangles after _repairMesh() — use halfedge traversal
        // to avoid allocating a std::vector per face.
        auto h = mesh_.halfedge(f);
        const Point& p0 = mesh_.point(mesh_.target(h));
        h = mesh_.next(h);
        const Point& p1 = mesh_.point(mesh_.target(h));
        h = mesh_.next(h);
        const Point& p2 = mesh_.point(mesh_.target(h));
        centroids_.push_back(CGAL::centroid(p0, p1, p2));
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

void Scene::_buildNormals() const
{
    normals_.clear();
    normals_.reserve(mesh_.number_of_faces());

    for (auto f : mesh_.faces()) {
        Vector n = PMP::compute_face_normal(f, mesh_);
        normals_.push_back(normalizedOrZero(n));
    }

    normalsDirty_ = false;
    spdlog::debug("_buildNormals: {} normals", normals_.size());
}

const std::vector<Vector>& Scene::faceNormals() const
{
    if (normalsDirty_)
        _buildNormals();
    return normals_;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray tracing
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> Scene::traceRays(const Point& source) const
{
    std::vector<float> distances = rayTracer_->traceRay(source);

    if (distances.size() != mesh_.number_of_faces()) {
        spdlog::warn("traceRays: distance count ({}) != face count ({})",
                     distances.size(), mesh_.number_of_faces());
    }

    return distances;
}

// ─────────────────────────────────────────────────────────────────────────────
// Acoustic computation
// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> Scene::computeNoiseMap(const std::vector<float>& distances,
                                           const AcousticModel&      model,
                                           const Point&              source) const
{
    const auto t_start = std::chrono::high_resolution_clock::now();

    const size_t N = distances.size();
    const float visibility_threshold =
        std::numeric_limits<float>::max() * 0.5f;
    std::vector<double> spl(N, -std::numeric_limits<double>::infinity());

    const auto& ap     = model.params();
    const double scale = ap.unit_scale;

    const double hs_m = source.z() * scale;
    const auto& cents = faceCentroids();
    const auto& norms = faceNormals();

    // ── Pre-compute spectral constants (invariant across all faces) ──────
    double rpm_eff = 0.0;
    if (ap.drone_model) {
        rpm_eff = ap.rpm_actual;
        if (rpm_eff <= 0.0) {
            ManoeuvreParams mp = AcousticModel::getManoeuvreParams(
                ap.manoeuvre, *ap.drone_model);
            rpm_eff = mp.rpm_average;
        }
    }

    double Lw_pre[NUM_BANDS];
    if (ap.drone_model) {
        AcousticModel::computeDroneLw(*ap.drone_model, rpm_eff, Lw_pre, ap.d_ref);
    } else {
        for (int b = 0; b < NUM_BANDS; ++b)
            Lw_pre[b] = ap.source_Lw[b];
    }

    // Absorption coefficients [dB/m] — depend only on atmosphere, constant
    double abs_coeff[NUM_BANDS];
    {
        AcousticModel base_model(ap);
        for (int b = 0; b < NUM_BANDS; ++b)
            abs_coeff[b] = base_model.absorptionCoefficient(THIRD_OCTAVE_FREQS[b]);
    }

    // Pre-compute Lw + A-weighting combined (avoids addition in hot loop)
    double Lw_Aw[NUM_BANDS];
    for (int b = 0; b < NUM_BANDS; ++b)
        Lw_Aw[b] = Lw_pre[b] + A_WEIGHTING[b];

    const double neg_inf = -std::numeric_limits<double>::infinity();

    // Fast inline SPL computation (direct_only mode: no ground effect).
    // Replaces per-face AcousticModel construction + computeSPL call.
    auto fastSPL = [&](double distance_m, double receiver_height_m) -> double {
        if (distance_m <= 0.0) return neg_inf;
        const double dz = hs_m - receiver_height_m;
        const double d_horiz = std::sqrt(std::max(0.0,
            distance_m * distance_m - dz * dz));
        const double theta = AcousticModel::elevationAngle(d_horiz, dz);
        const double A_div = AcousticModel::geometricalSpreading(distance_m);

        double sum = 0.0;
        for (int b = 0; b < NUM_BANDS; ++b) {
            const double D = AcousticModel::directivityCorrection(
                theta, THIRD_OCTAVE_FREQS[b]);
            const double Lp_Aw = Lw_Aw[b] - A_div
                - abs_coeff[b] * distance_m + D;
            sum += std::exp(Lp_Aw * LN10_OVER_10);
        }
        return 10.0 * std::log10(std::max(sum, 1e-30));
    };

    struct ReflectionCandidate {
        bool   valid = false;
        size_t reflector_idx = 0;
        double total_distance_m = 0.0;
        double spl_db = -std::numeric_limits<double>::infinity();
        double alignment_cos = -1.0;
    };

    std::vector<ReflectionCandidate> best_reflection(N);

    size_t reflector_count = 0;
    size_t tested_pairs = 0;
    size_t valid_pairs = 0;
    size_t rejected_invalid_normal = 0;
    size_t rejected_reflector_not_visible = 0;
    size_t rejected_target_not_visible = 0;
    size_t rejected_zero_reflected_dir = 0;
    size_t rejected_zero_outgoing_dir = 0;
    size_t rejected_alignment = 0;

    if (ap.reflection_order >= 1) {
        const double eps_mesh = std::max(1e-6, 0.01 / std::max(scale, 1e-9));
        const double min_alignment_cos = std::cos(15.0 * M_PI / 180.0);

        for (size_t j = 0; j < N && j < cents.size() && j < norms.size(); ++j) {
            const float d_source_to_reflector_raw = distances[j];
            if (d_source_to_reflector_raw <= 0.0f
                || d_source_to_reflector_raw >= visibility_threshold) {
                ++rejected_reflector_not_visible;
                continue;
            }

            const Point& reflector = cents[j];
            // Normals are already unit-length from _buildNormals();
            // skip only if the stored normal was a zero vector (degenerate face).
            Vector normal = norms[j];
            if (normal.squared_length() < 0.5) {
                ++rejected_invalid_normal;
                continue;
            }

            ++reflector_count;

            Vector to_source = source - reflector;
            if (dotProduct(normal, to_source) < 0.0)
                normal = -normal;

            const Point offset_origin = reflector + eps_mesh * normal;
            const auto reflected_distances = rayTracer_->traceRay(offset_origin);

            const double d_source_to_reflector_m =
                std::max(static_cast<double>(d_source_to_reflector_raw) * scale,
                         1e-3);

            const Vector incident_dir = normalizedOrZero(reflector - source);
            const Vector reflected_dir_expected =
                normalizedOrZero(reflectAroundNormal(incident_dir, normal));

            if (reflected_dir_expected.squared_length() < 1e-24) {
                ++rejected_zero_reflected_dir;
                continue;
            }

            for (size_t i = 0; i < N && i < cents.size(); ++i) {
                if (i == j || i >= reflected_distances.size())
                    continue;

                const float d_reflector_to_target_raw = reflected_distances[i];
                if (d_reflector_to_target_raw <= 0.0f
                    || d_reflector_to_target_raw >= visibility_threshold) {
                    ++rejected_target_not_visible;
                    continue;
                }

                ++tested_pairs;

                const Point& target = cents[i];
                const Vector outgoing_dir = normalizedOrZero(target - reflector);
                if (outgoing_dir.squared_length() < 1e-24) {
                    ++rejected_zero_outgoing_dir;
                    continue;
                }

                const double alignment_cos =
                    dotProduct(reflected_dir_expected, outgoing_dir);
                if (alignment_cos < min_alignment_cos) {
                    ++rejected_alignment;
                    continue;
                }

                const double dx_rt = (target.x() - reflector.x()) * scale;
                const double dy_rt = (target.y() - reflector.y()) * scale;
                const double dz_rt = (target.z() - reflector.z()) * scale;
                const double d_reflector_to_target_m =
                    std::max(std::sqrt(dx_rt * dx_rt + dy_rt * dy_rt + dz_rt * dz_rt),
                             1e-3);

                const double total_distance_m =
                    d_source_to_reflector_m + d_reflector_to_target_m;
                const double hr_target_m = target.z() * scale;
                const double candidate_spl =
                    fastSPL(total_distance_m, hr_target_m);

                if (!std::isfinite(candidate_spl))
                    continue;

                ++valid_pairs;

                ReflectionCandidate& best = best_reflection[i];
                if (!best.valid || candidate_spl > best.spl_db) {
                    best.valid = true;
                    best.reflector_idx = j;
                    best.total_distance_m = total_distance_m;
                    best.spl_db = candidate_spl;
                    best.alignment_cos = alignment_cos;
                }
            }
        }
    }

    double spl_max = -std::numeric_limits<double>::infinity();
    double spl_min =  std::numeric_limits<double>::infinity();
    size_t vis_direct    = 0;
    size_t vis_reflected = 0;

    for (size_t i = 0; i < N; ++i) {
        const float d_raw        = distances[i];
        const bool direct_visible =
            (d_raw > 0.0f && d_raw < visibility_threshold);

        if (i >= cents.size()) continue;
        const Point& c = cents[i];

        const double hr_m = c.z() * scale;

        double spl_direct = neg_inf;
        double spl_reflected = best_reflection[i].valid
            ? best_reflection[i].spl_db
            : neg_inf;

        if (direct_visible) {
            const double d_direct_m = std::max(
                static_cast<double>(d_raw) * scale, 1e-3);

            spl_direct = fastSPL(d_direct_m, hr_m);
            spl[i] = energySumDb(spl_direct, spl_reflected);
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

    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    spdlog::info("computeNoiseMap: {} faces, {}/{} direct, {}/{} reflected-only, "
                 "{} reflectors, {} valid specular paths, "
                 "SPL [{:.1f}, {:.1f}] dB(A) — {:.1f} s",
                 N, vis_direct, N, vis_reflected, N,
                 reflector_count, valid_pairs,
                 std::isfinite(spl_min) ? spl_min : 0.0,
                 std::isfinite(spl_max) ? spl_max : 0.0,
                 elapsed_ms / 1000.0);

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
                double energy = std::exp(spl[i] * LN10_OVER_10);
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

    // Les sommets qui ne bordent AUCUNE face avec un SPL fini restent
    // au gris par défaut (vCount == 0 → pas touché par la boucle ci-dessus).
    // On ne réinitialise PAS les sommets partagés entre faces visibles et
    // occultées : cela écrasait la couleur des faces visibles en bordure
    // d'ombre et créait des « trous » angulaires dans la noise map.

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
    if (bboxDirty_) {
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

        bboxMin_ = Point(xmin, ymin, zmin);
        bboxMax_ = Point(xmax, ymax, zmax);
        bboxDirty_ = false;
    }

    bbox_min = bboxMin_;
    bbox_max = bboxMax_;

    return p.x() >= bboxMin_.x() - margin && p.x() <= bboxMax_.x() + margin
        && p.y() >= bboxMin_.y() - margin && p.y() <= bboxMax_.y() + margin
        && p.z() >= bboxMin_.z() - margin && p.z() <= bboxMax_.z() + margin;
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
