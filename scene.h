/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#pragma once

#include "acoustic_model.h"
#include "ray_tracer.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/centroid.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>

#include <string>
#include <vector>
#include <array>
#include <limits>
#include <cstdint>

// ─── CGAL types ───────────────────────────────────────────────────────────────

using Kernel      = CGAL::Simple_cartesian<double>;
using Point       = Kernel::Point_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

// ─── Colour helpers ───────────────────────────────────────────────────────────

struct RGB {
    uint8_t r, g, b;
};

/// Jet colormap: blue (min) → cyan → green → yellow → red (max)
RGB splToJetColor(double spl, double spl_min, double spl_max);

// ─── Scene ────────────────────────────────────────────────────────────────────

class Scene {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /// Load mesh from a PLY file, apply basic mesh repair, build OptiX BVH.
    explicit Scene(const std::string& ply_path);

    /// Destructor – frees GPU resources
    ~Scene();

    // Non-copyable
    Scene(const Scene&)            = delete;
    Scene& operator=(const Scene&) = delete;

    // ── Ray tracing ──────────────────────────────────────────────────────────

    /// Trace one ray per face from @p source; returns distance per face
    /// (std::numeric_limits<float>::max() when occluded / no hit).
    std::vector<float> traceRays(const Point& source) const;

    // ── Acoustic computation ─────────────────────────────────────────────────

    /// Compute A-weighted SPL [dB(A)] for every face given ray distances.
    /// Includes ground reflection for occluded faces (image source method).
    /// @param distances  Per-face distance from direct ray tracing
    /// @param model      Acoustic propagation model
    /// @param source     3D position of the sound source
    std::vector<double> computeNoiseMap(const std::vector<float>&  distances,
                                        const AcousticModel&       model,
                                        const Point&               source) const;

    // ── Property attachment ──────────────────────────────────────────────────

    /// Attach raw ray distances as a face property ("f:distance").
    void addDistances(const std::vector<float>& distances);

    /// Attach A-weighted SPL values as a face property ("f:spl_dBA").
    void addSPL(const std::vector<double>& spl);

    /// Compute and attach per-vertex RGB colours from the SPL map ("v:color").
    void addNoiseMapColor(const std::vector<double>& spl);

    // ── I/O ──────────────────────────────────────────────────────────────────

    /// Write the mesh (geometry + all attached properties) to a PLY file.
    void writeMeshToPLY(const std::string& out_path) const;

    // ── Accessors ────────────────────────────────────────────────────────────

    const SurfaceMesh& mesh()       const { return mesh_; }
    std::size_t        numFaces()   const { return mesh_.number_of_faces(); }
    std::size_t        numVertices()const { return mesh_.number_of_vertices(); }

    /// Face centroids (computed once, cached).
    const std::vector<Point>& faceCentroids() const;

    /// Returns true if the point lies within the mesh bounding box (with margin).
    /// Sets bbox_min / bbox_max to the mesh bounds.
    bool isInsideBBox(const Point& p, double margin,
                      Point& bbox_min, Point& bbox_max) const;

private:
    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Load PLY into mesh_; throws on failure.
    void _loadPLY(const std::string& path);

    /// Basic mesh repair: remove degenerate faces, stitch borders.
    void _repairMesh();

    /// Build face-centroid cache.
    void _buildCentroids() const;

    // ── Data members ─────────────────────────────────────────────────────────

    SurfaceMesh                   mesh_;
    std::unique_ptr<RayTracer>    rayTracer_;
    mutable std::vector<Point>    centroids_;   ///< cached face centroids
    mutable bool                  centroidsDirty_ = true;
};
