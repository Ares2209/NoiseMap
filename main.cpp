/*
 * Created on Fri Oct 03 2025
 *
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0 (the 'License')
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * **Additional Restriction**: This code may not be used for commercial purposes.
 */

#include "scene.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <string>
#include <iostream>
#include <stdexcept>

// ─── Logger ──────────────────────────────────────────────────────────────────

void setupLogger()
{
    auto console_sink =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[console %^%l%$] %v");

    auto file_sink =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>("ray_tracer.log", true);
    file_sink->set_level(spdlog::level::trace);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

    std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
    auto logger = std::make_shared<spdlog::logger>(
        "multi_sink", sinks.begin(), sinks.end());

    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::trace);
}

// ─── Usage ───────────────────────────────────────────────────────────────────

void printUsage(const char* progname)
{
    std::cerr
        << "Usage: " << progname
        << " <ply_file> <x> <y> <z>"
        << " [--ground asphalt|soil|grass]"
        << " [--temp <celsius>]"
        << " [--humidity <percent>]"
        << " [--source-height <meters>]"
        << " [--receiver-height <meters>]\n\n"
        << "Computes a noise map (dBA) on the mesh from a point source at (x,y,z).\n"
        << "Acoustic model:\n"
        << "  - Geometrical spreading  (1/r²)\n"
        << "  - Atmospheric absorption (ISO 9613-1)\n"
        << "  - Vertical radiation directivity\n"
        << "  - Ground reflection      (Delany-Bazley)\n\n";
}

// ─── Argument parsing ────────────────────────────────────────────────────────

bool parseArguments(int argc, char* argv[],
                    std::string&    plyFilePath,
                    Point&          emission,
                    AcousticParams& ap)
{
    if (argc < 5) {
        printUsage(argv[0]);
        return false;
    }

    plyFilePath = argv[1];
    emission    = Point(std::stod(argv[2]),
                        std::stod(argv[3]),
                        std::stod(argv[4]));

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--ground" && i + 1 < argc) {
            std::string gt = argv[++i];
            if      (gt == "asphalt") ap.ground_type = GroundType::ASPHALT;
            else if (gt == "soil")    ap.ground_type = GroundType::COMPACT_SOIL;
            else if (gt == "grass")   ap.ground_type = GroundType::GRASS;
            else {
                spdlog::error("Unknown ground type '{}'. "
                              "Valid: asphalt | soil | grass", gt);
                return false;
            }
        }
        else if (arg == "--temp" && i + 1 < argc) {
            ap.temperature_C = std::stod(argv[++i]);
        }
        else if (arg == "--humidity" && i + 1 < argc) {
            ap.humidity_pct = std::stod(argv[++i]);
        }
        else if (arg == "--source-height" && i + 1 < argc) {
            ap.source_height = std::stod(argv[++i]);
        }
        else if (arg == "--receiver-height" && i + 1 < argc) {
            ap.receiver_height = std::stod(argv[++i]);
        }
        else {
            spdlog::error("Unknown argument: {}", arg);
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    setupLogger();

    std::string    plyFilePath;
    Point          emission;
    AcousticParams acousticParams;

    if (!parseArguments(argc, argv, plyFilePath, emission, acousticParams))
        return EXIT_FAILURE;

    spdlog::info("PLY file      : {}", plyFilePath);
    spdlog::info("Source        : ({:.3f}, {:.3f}, {:.3f})",
                 emission.x(), emission.y(), emission.z());
    spdlog::info("Acoustic      : T={:.1f}°C  RH={:.0f}%  "
                 "src_h={:.2f}m  rcv_h={:.2f}m",
                 acousticParams.temperature_C,
                 acousticParams.humidity_pct,
                 acousticParams.source_height,
                 acousticParams.receiver_height);

    try {
        // ── 1. Load scene (mesh loading + repair centralisés dans Scene) ──
        Scene scene(plyFilePath);   // utilise Scene(const std::string&)

        // ── 2. Ray tracing ────────────────────────────────────────────────
        std::vector<float> distances = scene.traceRays(emission);
        scene.addDistances(distances);

        // ── 3. Acoustic model ─────────────────────────────────────────────
        AcousticModel            model(acousticParams);
        std::vector<double> spl_dBA = scene.computeNoiseMap(distances, model);
        scene.addSPL(spl_dBA);

        // ── 4. Colorisation ───────────────────────────────────────────────
        scene.addNoiseMapColor(spl_dBA);

        // ── 5. Export PLY ─────────────────────────────────────────────────
        const std::string outPath =
            plyFilePath.substr(0, plyFilePath.find_last_of('.'))
            + "_noisemap.ply";

        scene.writeMeshToPLY(outPath);
        spdlog::info("Output written to '{}'", outPath);
    }
    catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
