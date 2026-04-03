/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#pragma once
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Third-octave centre frequencies (29 bands, 25 Hz … 20 kHz) ─────────────
static constexpr int NUM_BANDS = 29;

static constexpr double THIRD_OCTAVE_FREQS[NUM_BANDS] = {
     25,   31.5,   40,   50,   63,   80,  100,  125,  160,
    200,   250,   315,  400,  500,  630,  800, 1000, 1250,
   1600,  2000,  2500, 3150, 4000, 5000, 6300, 8000,10000,
  12500, 16000
};

// A-weighting corrections for each third-octave band [dB]
static constexpr double A_WEIGHTING[NUM_BANDS] = {
    -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4,
    -10.9,  -8.6,  -6.6,  -4.8,  -3.2,  -1.9,  -0.8,   0.0,   0.6,
      1.0,   1.2,   1.3,   1.2,   1.0,   0.5,  -0.1,  -1.1,  -2.5,
     -4.3,  -6.6
};

// ─── Ground types ─────────────────────────────────────────────────────────────
enum class GroundType { ASPHALT, COMPACT_SOIL, GRASS };

// ─── Parameters ───────────────────────────────────────────────────────────────
struct AcousticParams {
    double source_Lw[NUM_BANDS] = {};  // Sound power level per band [dB re 1 pW]
    double source_height   = 50.0;     // Source height above ground [m]
    double receiver_height =  1.5;     // Receiver height above ground [m]
    double temperature_C   = 20.0;     // Air temperature [°C]
    double humidity_pct    = 70.0;     // Relative humidity [%]
    double pressure_kPa    = 101.325;  // Atmospheric pressure [kPa]
    GroundType ground_type = GroundType::GRASS;
};

// ─── AcousticModel ────────────────────────────────────────────────────────────
class AcousticModel {
public:
    explicit AcousticModel(const AcousticParams& params);

    // Compute per-band SPL at receiver
    void   computeSPLSpectrum(double distance_m, bool visible,
                               double out_bands[NUM_BANDS]) const;

    // Compute overall A-weighted SPL [dBA]
    double computeSPL(double distance_m, bool visible) const;

    // Individual sub-models (public for unit testing)
    static double geometricalSpreading(double distance_m);
    double        atmosphericAbsorption(double freq_Hz, double distance_m) const;
    static double directivityCorrection(double theta_deg, double freq_Hz);
    double        groundEffect(double freq_Hz, double distance_m) const;

    static double getGroundResistivity(GroundType type);

private:
    AcousticParams params_;
    double         ground_resistivity_;   // [Pa·s/m²]

    double absorptionCoefficient(double freq_Hz) const;

    // theta = elevation angle from horizontal plane [deg], negative = below source
    static double elevationAngle(double horizontal_distance_m, double dz_m);
};
