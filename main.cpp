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
#include "acoustic_model.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <string>
#include <iostream>
#include <stdexcept>
#include <chrono>

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
        << " <ply_file> <x> <y> <z>\n"
        << "\n"
        << "  Coordinate system: Z=0 is ground, Z>0 is buildings.\n"
        << "  Coordinates are in mesh units (1 unit = --scale meters).\n"
        << "\n"
        << "  [--scale         <meters>]          (1 mesh unit = N meters, default 100)\n"
        << "  [--reflections   <0|1>]             (0=direct only, 1=+ground reflection, default 1)\n"
        << "  [--reflect-filter <dB(A)>]          (skip reflection if estimated SPL < threshold)\n"
        << "  [--ground        asphalt|soil|grass]\n"
        << "  [--temp          <celsius>]\n"
        << "  [--humidity      <percent>]\n"
        << "  [--pressure      <kPa>]\n"
        << "  [--source-height <meters>]          (override, default = Z × scale)\n"
        << "  [--receiver-height <meters>]\n"
        << "\n"
        << "  Drone options (optional – overrides generic Lw if provided):\n"
        << "  [--drone         M2|I2|F-4|S-9]\n"
        << "  [--rpm           <r/min>]          (0 = automatic from manoeuvre)\n"
        << "  [--manoeuvre     hover10|hover20|hover50|climb|sink|fwd-down|fwd-up]\n"
        << "  [--wind          <m/s>]             (wind speed at 2.5 m)\n"
        << "  [--d-ref         <meters>]          (lab recording distance, default 1.5)\n"
        << "\n"
        << "Description:\n"
        << "  Computes a noise map (dBA) on the mesh from a point source at (x,y,z).\n"
        << "\n"
        << "Acoustic model:\n"
        << "  - Geometrical spreading   (1/r²)\n"
        << "  - Atmospheric absorption  (ISO 9613-1)\n"
        << "  - Vertical directivity    (polynomial + high-shelving filter)\n"
        << "  - Ground reflection       (Delany-Bazley, image source method) [if --reflections 1]\n"
        << "  - RPM equalisation        (Eq. 2, Heutschi et al. 2020) [if --drone]\n"
        << "  - Manoeuvre RPM model     (Tables 4-5)                  [if --drone]\n"
        << "\n"
        << "Available drone models:\n";

    for (const auto& m : DroneDatabase::instance().all()) {
        std::cerr
            << "    " << m.name
            << "  (" << m.num_rotors << " rotors, "
            << m.weight_g << " g, "
            << "R_ref=" << m.rpm_ref << " r/min)\n";
    }
    std::cerr << "\n";
}

// ─── Helpers de parsing ───────────────────────────────────────────────────────

/// Convertit une chaîne de manœuvre vers l'enum FlightManoeuvre.
/// Retourne false si la valeur est inconnue.
static bool parseManoeuvre(const std::string& s, FlightManoeuvre& out)
{
    if      (s == "hover10")   { out = FlightManoeuvre::HOVER_10M;          return true; }
    else if (s == "hover20")   { out = FlightManoeuvre::HOVER_20M;          return true; }
    else if (s == "hover50")   { out = FlightManoeuvre::HOVER_50M;          return true; }
    else if (s == "climb")     { out = FlightManoeuvre::CLIMB;              return true; }
    else if (s == "sink")      { out = FlightManoeuvre::SINK;               return true; }
    else if (s == "fwd-down")  { out = FlightManoeuvre::FORWARD_DOWNWIND;   return true; }
    else if (s == "fwd-up")    { out = FlightManoeuvre::FORWARD_UPWIND;     return true; }
    return false;
}

/// Retourne le nom lisible d'une manœuvre (pour les logs).
static std::string manoeuvreLabel(FlightManoeuvre m)
{
    switch (m) {
        case FlightManoeuvre::HOVER_10M:         return "hover @ 10 m";
        case FlightManoeuvre::HOVER_20M:         return "hover @ 20 m";
        case FlightManoeuvre::HOVER_50M:         return "hover @ 50 m";
        case FlightManoeuvre::CLIMB:             return "climb";
        case FlightManoeuvre::SINK:              return "sink";
        case FlightManoeuvre::FORWARD_DOWNWIND:  return "forward downwind";
        case FlightManoeuvre::FORWARD_UPWIND:    return "forward upwind";
    }
    return "unknown";
}

// ─── Argument parsing ────────────────────────────────────────────────────────

/**
 * @brief Parse la ligne de commande et remplit plyFilePath, emission et ap.
 *
 * Les arguments de drone (--drone, --rpm, --manoeuvre, --wind, --d-ref)
 * sont stockés dans des variables locales et résolus après la boucle
 * pour valider la cohérence (ex. --rpm sans --drone → avertissement).
 *
 * @return true si le parsing réussit, false sinon.
 */
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
    try {
        emission = Point(std::stod(argv[2]),
                         std::stod(argv[3]),
                         std::stod(argv[4]));
    } catch (const std::exception&) {
        spdlog::error("Invalid source coordinates: {} {} {}",
                      argv[2], argv[3], argv[4]);
        return false;
    }

    // Valeurs intermédiaires pour les options drone
    std::string droneName;
    double      rpm_cli    = 0.0;
    double      wind_cli   = 2.0;
    double      d_ref_cli  = 1.5;
    FlightManoeuvre mano   = FlightManoeuvre::HOVER_10M;

    // ── Boucle de parsing ─────────────────────────────────────────────────────
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];

        // ── Échelle & réflexion ───────────────────────────────────────────────
        if (arg == "--scale" && i + 1 < argc) {
            ap.unit_scale = std::stod(argv[++i]);
            if (ap.unit_scale <= 0.0) {
                spdlog::error("Scale must be > 0 (got {:.3f})", ap.unit_scale);
                return false;
            }
        }
        else if (arg == "--reflections" && i + 1 < argc) {
            ap.reflection_order = std::stoi(argv[++i]);
            if (ap.reflection_order < 0 || ap.reflection_order > 1) {
                spdlog::error("Reflection order must be 0 or 1 (got {})",
                              ap.reflection_order);
                return false;
            }
        }
        else if (arg == "--reflect-filter" && i + 1 < argc) {
            ap.reflect_filter = std::stod(argv[++i]);
        }
        // ── Options générales ─────────────────────────────────────────────────
        else if (arg == "--ground" && i + 1 < argc) {
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
        else if (arg == "--pressure" && i + 1 < argc) {
            ap.pressure_kPa = std::stod(argv[++i]);
        }
        else if (arg == "--source-height" && i + 1 < argc) {
            ap.source_height = std::stod(argv[++i]);
        }
        else if (arg == "--receiver-height" && i + 1 < argc) {
            ap.receiver_height = std::stod(argv[++i]);
        }

        // ── Options drone ─────────────────────────────────────────────────────
        else if (arg == "--drone" && i + 1 < argc) {
            droneName = argv[++i];
        }
        else if (arg == "--rpm" && i + 1 < argc) {
            rpm_cli = std::stod(argv[++i]);
            if (rpm_cli < 0.0) {
                spdlog::error("RPM must be >= 0 (got {:.1f})", rpm_cli);
                return false;
            }
        }
        else if (arg == "--manoeuvre" && i + 1 < argc) {
            if (!parseManoeuvre(argv[++i], mano)) {
                spdlog::error("Unknown manoeuvre '{}'. "
                              "Valid: hover10|hover20|hover50|"
                              "climb|sink|fwd-down|fwd-up", argv[i]);
                return false;
            }
        }
        else if (arg == "--wind" && i + 1 < argc) {
            wind_cli = std::stod(argv[++i]);
            if (wind_cli < 0.0) {
                spdlog::error("Wind speed must be >= 0 (got {:.2f})", wind_cli);
                return false;
            }
        }
        else if (arg == "--d-ref" && i + 1 < argc) {
            d_ref_cli = std::stod(argv[++i]);
            if (d_ref_cli <= 0.0) {
                spdlog::error("Reference distance must be > 0 (got {:.3f})", d_ref_cli);
                return false;
            }
        }
        else {
            spdlog::error("Unknown or incomplete argument: '{}'", arg);
            printUsage(argv[0]);
            return false;
        }
    }

    // ── source_height auto depuis Z × scale si non fourni explicitement ─────
    if (ap.source_height < 0.0)
        ap.source_height = emission.z() * ap.unit_scale;

    // ── Résolution du modèle de drone ─────────────────────────────────────────
    if (!droneName.empty()) {
        const DroneEmissionModel* dronePtr =
            DroneDatabase::instance().find(droneName);

        if (!dronePtr) {
            spdlog::error("Unknown drone model '{}'. Available: M2, I2, F-4, S-9",
                          droneName);
            return false;
        }

        ap.drone_model      = dronePtr;
        ap.rpm_actual       = rpm_cli;
        ap.manoeuvre        = mano;
        ap.wind_speed_2_5m  = wind_cli;
        ap.d_ref            = d_ref_cli;

    } else {
        // Avertissements si des options drone sont passées sans --drone
        if (rpm_cli > 0.0)
            spdlog::warn("--rpm ignored: no --drone model specified");
        if (wind_cli != 2.0)
            spdlog::warn("--wind ignored: no --drone model specified");
        if (d_ref_cli != 1.5)
            spdlog::warn("--d-ref ignored: no --drone model specified");

        ap.drone_model = nullptr;
    }

    return true;
}

// ─── Logging du résumé des paramètres ────────────────────────────────────────

static void logParameters(const Point&          emission,
                           const AcousticParams& ap)
{
    spdlog::info("─── Référentiel ───────────────────────────────────────");
    spdlog::info("  Scale         : 1 unit = {:.1f} m", ap.unit_scale);
    spdlog::info("  Reflections   : {}", ap.reflection_order);
    if (std::isfinite(ap.reflect_filter))
        spdlog::info("  Reflect filter: {:.1f} dB(A)", ap.reflect_filter);
    else
        spdlog::info("  Reflect filter: off (all reflections computed)");

    spdlog::info("─── Source ────────────────────────────────────────────");
    spdlog::info("  Position mesh : ({:.4f}, {:.4f}, {:.4f})",
                 emission.x(), emission.y(), emission.z());
    spdlog::info("  Position [m]  : ({:.1f}, {:.1f}, {:.1f})",
                 emission.x() * ap.unit_scale,
                 emission.y() * ap.unit_scale,
                 emission.z() * ap.unit_scale);
    spdlog::info("  Source height : {:.2f} m  (Z=0 = sol)", ap.source_height);

    spdlog::info("─── Atmosphere ────────────────────────────────────────");
    spdlog::info("  Temperature   : {:.1f} °C",  ap.temperature_C);
    spdlog::info("  Humidity      : {:.0f} %",   ap.humidity_pct);
    spdlog::info("  Pressure      : {:.3f} kPa", ap.pressure_kPa);

    spdlog::info("─── Ground ────────────────────────────────────────────");
    const char* gt = (ap.ground_type == GroundType::ASPHALT)      ? "asphalt"
                   : (ap.ground_type == GroundType::COMPACT_SOIL)  ? "compact soil"
                                                                    : "grass";
    spdlog::info("  Ground type   : {}", gt);

    if (ap.drone_model) {
        const DroneEmissionModel& dm = *ap.drone_model;

        spdlog::info("─── Drone ─────────────────────────────────────────────");
        spdlog::info("  Model         : {}", dm.name);
        spdlog::info("  Rotors        : {}",     dm.num_rotors);
        spdlog::info("  Weight        : {:.0f} g", dm.weight_g);
        spdlog::info("  R_ref         : {:.0f} r/min", dm.rpm_ref);
        spdlog::info("  Manoeuvre     : {}", manoeuvreLabel(ap.manoeuvre));
        spdlog::info("  Wind speed    : {:.1f} m/s", ap.wind_speed_2_5m);
        spdlog::info("  d_ref         : {:.2f} m",   ap.d_ref);

        // RPM effectif
        double rpm_eff = ap.rpm_actual;
        if (rpm_eff <= 0.0) {
            ManoeuvreParams mp =
                AcousticModel::getManoeuvreParams(ap.manoeuvre, dm);
            rpm_eff = mp.rpm_average;
        }
        spdlog::info("  RPM (eff.)    : {:.0f} r/min{}",
                     rpm_eff,
                     (ap.rpm_actual <= 0.0) ? "  [from manoeuvre table]" : "");

        // Déviation standard du RPM
        double sigma_n = AcousticModel::rpmNormalisedStdDev(
            ap.manoeuvre, ap.wind_speed_2_5m, dm);
        spdlog::info("  σn (RPM std)  : {:.4f}", sigma_n);

    } else {
        spdlog::info("─── Drone ─────────────────────────────────────────────");
        spdlog::info("  Model         : (generic – using source_Lw[])");
    }

    spdlog::info("───────────────────────────────────────────────────────");
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

    spdlog::info("PLY file : {}", plyFilePath);
    logParameters(emission, acousticParams);

    try {
        using clk = std::chrono::high_resolution_clock;
        auto t0 = clk::now();

        // ── 1. Chargement de la scène ─────────────────────────────────────
        Scene scene(plyFilePath);
        auto t1 = clk::now();

        // ── 2. Ray tracing ────────────────────────────────────────────────
        std::vector<float> distances = scene.traceRays(emission);
        scene.addDistances(distances);
        auto t2 = clk::now();

        // ── 3. Modèle acoustique (direct + réflexion au sol) ────────────
        AcousticModel       model(acousticParams);
        std::vector<double> spl_dBA = scene.computeNoiseMap(distances, model,
                                                             emission);
        scene.addSPL(spl_dBA);
        auto t3 = clk::now();

        // ── 4. Colorisation ───────────────────────────────────────────────
        scene.addNoiseMapColor(spl_dBA);
        auto t4 = clk::now();

        // ── 5. Export PLY ─────────────────────────────────────────────────
        const std::string outPath =
            plyFilePath.substr(0, plyFilePath.find_last_of('.'))
            + "_noisemap.ply";

        scene.writeMeshToPLY(outPath);
        auto t5 = clk::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        spdlog::info("─── Timing ────────────────────────────────────────────");
        spdlog::info("  Load + repair : {} ms", ms(t0, t1));
        spdlog::info("  Ray tracing   : {} ms", ms(t1, t2));
        spdlog::info("  Acoustic      : {} ms", ms(t2, t3));
        spdlog::info("  Colorisation  : {} ms", ms(t3, t4));
        spdlog::info("  PLY write     : {} ms", ms(t4, t5));
        spdlog::info("  TOTAL         : {} ms", ms(t0, t5));

        spdlog::info("Output written to '{}'", outPath);

        // ── 6. Statistiques SPL ───────────────────────────────────────────
        double spl_min =  std::numeric_limits<double>::infinity();
        double spl_max = -std::numeric_limits<double>::infinity();
        double spl_sum =  0.0;
        int    spl_cnt =  0;

        for (double v : spl_dBA) {
            if (std::isfinite(v)) {
                spl_min  = std::min(spl_min, v);
                spl_max  = std::max(spl_max, v);
                spl_sum += v;
                ++spl_cnt;
            }
        }

        if (spl_cnt > 0) {
            spdlog::info("─── SPL statistics ({} vertices) ───────────────────",
                         spl_cnt);
            spdlog::info("  Min  : {:.1f} dB(A)", spl_min);
            spdlog::info("  Max  : {:.1f} dB(A)", spl_max);
            spdlog::info("  Mean : {:.1f} dB(A)", spl_sum / spl_cnt);
        }
    }
    catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
