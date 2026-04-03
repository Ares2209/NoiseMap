/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#include "acoustic_model.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Construction ─────────────────────────────────────────────────────────────

AcousticModel::AcousticModel(const AcousticParams& params)
    : params_(params)
    , ground_resistivity_(getGroundResistivity(params.ground_type))
{}

// ─── Ground resistivity ───────────────────────────────────────────────────────
// Values in Pa·s/m² (rayls/m) – standard Delany-Bazley flow resistivity.
// References: Delany & Bazley (1970), ISO 9613-2 Annex B.
double AcousticModel::getGroundResistivity(GroundType type)
{
    switch (type) {
        case GroundType::ASPHALT:      return 2.0e7;   // ≈ 20 MRayl/m (very hard)
        case GroundType::COMPACT_SOIL: return 3.0e5;   // ≈ 300 kRayl/m
        case GroundType::GRASS:        return 2.0e4;   // ≈  20 kRayl/m (absorptive)
    }
    return 2.0e4;
}

// ─── Geometrical spreading ────────────────────────────────────────────────────

double AcousticModel::geometricalSpreading(double distance_m)
{
    if (distance_m <= 0.0) return 0.0;
    // Free-field point source: Lp = Lw − 20·log₁₀(r) − 11 dB
    // → spreading attenuation = 20·log₁₀(r) + 10·log₁₀(4π)
    return 20.0 * std::log10(distance_m) + 10.0 * std::log10(4.0 * M_PI);
}

// ─── Atmospheric absorption (ISO 9613-1) ──────────────────────────────────────

double AcousticModel::absorptionCoefficient(double freq_Hz) const
{
    const double T     = params_.temperature_C + 273.15;  // [K]
    const double T_ref = 293.15;                           // 20 °C reference [K]
    const double T_rel = T / T_ref;
    const double p_rel = params_.pressure_kPa / 101.325;  // normalised pressure

    const double h = params_.humidity_pct;                 // [%]

    // Molar concentration of water vapour
    double C      = -6.8346 * std::pow(273.16 / T, 1.261) + 4.6151;
    double h_mol  = h * std::pow(10.0, C) * p_rel;         // [%]

    // Relaxation frequencies
    double fr_O = p_rel * (24.0 + 4.04e4 * h_mol *
                  (0.02 + h_mol) / (0.391 + h_mol));

    double fr_N = p_rel * std::pow(T_rel, -0.5) *
                  (9.0 + 280.0 * h_mol *
                   std::exp(-4.170 * (std::pow(T_rel, -1.0/3.0) - 1.0)));

    const double f2 = freq_Hz * freq_Hz;

    double alpha = 8.686 * f2 * (
          1.84e-11 / p_rel * std::sqrt(T_rel)
        + std::pow(T_rel, -2.5) * (
              0.01275 * std::exp(-2239.1 / T) / (fr_O + f2 / fr_O)
            + 0.1068  * std::exp(-3352.0 / T) / (fr_N + f2 / fr_N)
          )
    );

    return alpha;  // [dB/m]
}

double AcousticModel::atmosphericAbsorption(double freq_Hz, double distance_m) const
{
    return absorptionCoefficient(freq_Hz) * distance_m;
}

// ─── Directivity correction ───────────────────────────────────────────────────

double AcousticModel::directivityCorrection(double theta_deg, double freq_Hz)
{
    // Second-order polynomial fit to multi-rotor drone directivity.
    // θ = elevation angle from horizontal rotor plane (−90 … +90°).
    // G(θ) [dB]: positive = gain, negative = attenuation.
    double abs_theta = std::min(std::abs(theta_deg), 90.0);
    double G = -0.0011 * theta_deg * theta_deg + 0.194 * abs_theta - 4.9;

    // Frequency shaping: effect is reduced below 500 Hz
    if (freq_Hz < 500.0)
        G *= freq_Hz / 500.0;

    return G;
}

// ─── Elevation angle ──────────────────────────────────────────────────────────

double AcousticModel::elevationAngle(double horiz_m, double dz_m)
{
    // dz_m = source_height − receiver_height
    // Positive dz → source is above receiver → θ < 0 (receiver looks up, rotor points down)
    // Convention: θ measured from horizontal plane, negative = receiver is below source.
    if (horiz_m < 1e-9 && std::abs(dz_m) < 1e-9) return 0.0;

    // Angle from horizontal: atan2(vertical_drop, horizontal_distance)
    // We define θ as the angle seen from the *source* looking toward *receiver*,
    // measured from horizontal:
    //   θ = atan2(-(dz_m), horiz_m)   [negative because receiver is below]
    double theta = std::atan2(-dz_m, horiz_m) * (180.0 / M_PI);
    return std::clamp(theta, -90.0, 90.0);
}

// ─── Ground effect (Delany-Bazley + interference) ─────────────────────────────

double AcousticModel::groundEffect(double freq_Hz, double distance_m) const
{
    if (distance_m <= 0.0) return 0.0;

    const double hs = params_.source_height;    // source  height [m]
    const double hr = params_.receiver_height;  // receiver height [m]

    // Horizontal distance derived from slant distance and heights
    double dz     = hs - hr;
    double d_horiz = std::sqrt(std::max(0.0, distance_m * distance_m - dz * dz));

    // Direct path (3-D slant distance, already = distance_m)
    double d_direct = distance_m;

    // Reflected path (image source method)
    double d_reflect = std::sqrt(d_horiz * d_horiz + (hs + hr) * (hs + hr));

    // Path length difference
    double delta_d = d_reflect - d_direct;

    // ── Delany-Bazley normalised surface impedance ────────────────────────────
    // Z/Z₀ = 1 + 9.08·(f/σ)^−0.75 − j·11.9·(f/σ)^−0.73
    double f_over_sigma = freq_Hz / ground_resistivity_;

    // Guard against extreme values (very hard or very soft ground)
    f_over_sigma = std::clamp(f_over_sigma, 1e-10, 1.0);

    double Z_real = 1.0 + 9.08  * std::pow(f_over_sigma, -0.75);
    double Z_imag = -11.9 * std::pow(f_over_sigma, -0.73);  // negative imaginary part
    std::complex<double> Z_norm(Z_real, Z_imag);

    // ── Plane-wave reflection coefficient ─────────────────────────────────────
    // Grazing angle ψ: sin(ψ) = (hs + hr) / d_reflect
    double sin_psi = (hs + hr) / d_reflect;
    sin_psi = std::clamp(sin_psi, 0.0, 1.0);

    std::complex<double> Zsin = Z_norm * sin_psi;
    std::complex<double> R_p  = (Zsin - 1.0) / (Zsin + 1.0);

    // Amplitude ratio (geometric spreading of reflected vs direct)
    double Q_mag    = std::abs(R_p);
    double amp_ratio = Q_mag * (d_direct / d_reflect);

    // ── Interference ──────────────────────────────────────────────────────────
    double k           = 2.0 * M_PI * freq_Hz / 343.0;   // wavenumber [rad/m]
    double phase_refl  = std::arg(R_p);
    double total_phase = k * delta_d + phase_refl;

    // |p_direct + p_reflected|² / |p_direct|²
    double interference = 1.0
        + amp_ratio * amp_ratio
        + 2.0 * amp_ratio * std::cos(total_phase);

    interference = std::max(interference, 1e-10);
    return 10.0 * std::log10(interference);  // [dB]
}

// ─── SPL computation ──────────────────────────────────────────────────────────

void AcousticModel::computeSPLSpectrum(double distance_m, bool visible,
                                        double out_bands[NUM_BANDS]) const
{
    if (!visible || distance_m <= 0.0) {
        for (int i = 0; i < NUM_BANDS; ++i)
            out_bands[i] = -std::numeric_limits<double>::infinity();
        return;
    }

    // Elevation angle at receiver
    const double dz      = params_.source_height - params_.receiver_height;
    const double d_horiz = std::sqrt(std::max(0.0,
                           distance_m * distance_m - dz * dz));
    const double theta   = elevationAngle(d_horiz, dz);

    const double A_div = geometricalSpreading(distance_m);

    for (int i = 0; i < NUM_BANDS; ++i) {
        double f    = THIRD_OCTAVE_FREQS[i];
        double Lw   = params_.source_Lw[i];
        double A_atm = atmosphericAbsorption(f, distance_m);
        double D    = directivityCorrection(theta, f);
        double G    = groundEffect(f, distance_m);

        // Lp = Lw − A_div − A_atm + D + G
        out_bands[i] = Lw - A_div - A_atm + D + G;
    }
}

double AcousticModel::computeSPL(double distance_m, bool visible) const
{
    if (!visible || distance_m <= 0.0)
        return -std::numeric_limits<double>::infinity();

    double bands[NUM_BANDS];
    computeSPLSpectrum(distance_m, visible, bands);

    // Energy sum of A-weighted third-octave bands
    double sum = 0.0;
    for (int i = 0; i < NUM_BANDS; ++i) {
        double L_Aw = bands[i] + A_WEIGHTING[i];
        sum += std::pow(10.0, L_Aw / 10.0);
    }

    return 10.0 * std::log10(sum);
}
