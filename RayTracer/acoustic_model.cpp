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
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═════════════════════════════════════════════════════════════════════════════
// DroneDatabase  –  Base de données des modèles de drones
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Niveaux de pression de référence Lp [dB re 20 µPa @ 1.5 m, −30°]
 * lus approximativement sur la Figure 6 du papier pour chaque drone.
 *
 * Ordre des bandes : 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
 *                    1k, 1.25k, 1.6k, 2k, 2.5k, 3.15k, 4k, 5k,
 *                    6.3k, 8k, 10k, 12.5k, 16k, 20k  [Hz]
 */

// ── Mavic 2 Pro (M2) ──────────────────────────────────────────────────────────
static const std::array<double, NUM_BANDS> LP_REF_M2 = {
    40, 42, 43, 45, 55, 60, 58, 62, 65, 68,
    70, 68, 65, 63, 60, 58, 55, 52,
    48, 44, 40, 35, 28, 20
};

/// Pentes S [dB/(r/min)] pour M2, Tableau 3
static const std::array<double, NUM_BANDS> SLOPE_M2 = {
    1.5e-3, 2.9e-3, -1.3e-4, 4.4e-3, 3.8e-3, 4.6e-3, 4.4e-3, 4.3e-3,
    4.0e-3, 3.6e-3,  3.7e-3, 2.7e-3,  3.8e-3, 3.8e-3, 3.6e-3, 3.5e-3,
    3.5e-3, 3.3e-3,  3.3e-3, 3.3e-3,  2.3e-3, 7.5e-4,  1.8e-4, 1.8e-4
};

// ── Inspire 2 (I2) ────────────────────────────────────────────────────────────
static const std::array<double, NUM_BANDS> LP_REF_I2 = {
    45, 48, 50, 52, 60, 65, 63, 66, 68, 70,
    72, 70, 68, 65, 63, 60, 57, 54,
    50, 46, 42, 37, 30, 22
};

/// Pentes S [dB/(r/min)] pour I2, Tableau 3
static const std::array<double, NUM_BANDS> SLOPE_I2 = {
    6.6e-3, 5.4e-3, 9.8e-3, 1.1e-2, 8.7e-3, 9.3e-3, 1.1e-2, 8.1e-3,
    8.5e-3, 8.8e-3, 7.7e-3, 7.8e-3, 8.0e-3, 8.0e-3, 8.0e-3, 8.1e-3,
    8.4e-3, 8.4e-3, 9.0e-3, 8.3e-3, 7.9e-3, 6.2e-3, 4.9e-3, 4.9e-3
};

// ── F-450 (F-4) ───────────────────────────────────────────────────────────────
static const std::array<double, NUM_BANDS> LP_REF_F4 = {
    38, 40, 42, 44, 52, 57, 55, 58, 60, 63,
    65, 63, 60, 58, 55, 52, 50, 47,
    43, 39, 35, 30, 24, 16
};

/// Pentes S [dB/(r/min)] pour F-4, Tableau 3 (valide RPM > 6200 r/min)
static const std::array<double, NUM_BANDS> SLOPE_F4 = {
    2.3e-4, 3.6e-2, 0.0, 0.0, 3.3e-2, 1.7e-2, 1.2e-2, 4.1e-2,
    1.8e-2, 4.5e-2, 2.3e-2, 2.9e-2, 2.6e-2, 2.3e-2, 1.8e-2, 2.2e-2,
    1.8e-2, 2.0e-2, 2.0e-2, 1.4e-2, 1.7e-2, 1.9e-2, 1.9e-2, 1.9e-2
};

// ── S-900 (S-9) ───────────────────────────────────────────────────────────────
static const std::array<double, NUM_BANDS> LP_REF_S9 = {
    48, 50, 52, 55, 63, 68, 66, 70, 72, 74,
    75, 73, 70, 68, 65, 62, 59, 56,
    52, 48, 44, 39, 32, 24
};

/// Pentes S [dB/(r/min)] pour S-9, Tableau 3
static const std::array<double, NUM_BANDS> SLOPE_S9 = {
    5.8e-3, 1.4e-3, 4.1e-3, 7.4e-3, 6.3e-3, 1.0e-2, 7.0e-3, 8.2e-3,
    6.0e-3, 1.4e-3, 4.1e-3, 3.0e-3, 5.2e-4, 2.4e-3, 3.8e-3, 3.6e-3,
    4.3e-3, 4.0e-3, 3.8e-3, 4.3e-3, 2.8e-3, 7.4e-4, 2.9e-3, 2.9e-3
};

// ─── Construction de la base de données ──────────────────────────────────────

DroneDatabase::DroneDatabase()
{
    // Tableau 1 + Tableau 2 du papier
    models_ = {
        // ── Mavic 2 Pro ───────────────────────────────────────────────────────
        { "M2", 4, 907.0, 6540.0, LP_REF_M2, SLOPE_M2 },

        // ── Inspire 2 ─────────────────────────────────────────────────────────
        { "I2", 4, 3400.0, 4560.0, LP_REF_I2, SLOPE_I2 },

        // ── F-450 ─────────────────────────────────────────────────────────────
        { "F-4", 4, 800.0, 6420.0, LP_REF_F4, SLOPE_F4 },

        // ── S-900 ─────────────────────────────────────────────────────────────
        { "S-9", 6, 3300.0, 6900.0, LP_REF_S9, SLOPE_S9 },
    };

    // Construction de l'index
    for (size_t i = 0; i < models_.size(); ++i)
        index_[models_[i].name] = i;
}

const DroneDatabase& DroneDatabase::instance()
{
    static DroneDatabase db;
    return db;
}

const DroneEmissionModel* DroneDatabase::find(const std::string& name) const
{
    auto it = index_.find(name);
    return (it != index_.end()) ? &models_[it->second] : nullptr;
}

// ═════════════════════════════════════════════════════════════════════════════
// AcousticModel  –  Construction et utilitaires
// ═════════════════════════════════════════════════════════════════════════════

AcousticModel::AcousticModel(const AcousticParams& params)
    : params_(params)
    , ground_resistivity_(getGroundResistivity(params.ground_type))
{}

double AcousticModel::getGroundResistivity(GroundType type)
{
    // Valeurs en kPa·s/m² (convention Delany & Bazley 1970)
    // Le paramètre normalisé f/σ utilise σ en kPa·s/m² directement.
    switch (type) {
        case GroundType::ASPHALT:      return 20000.0; // 20 000 kPa·s/m²
        case GroundType::COMPACT_SOIL: return  5000.0; //  5 000 kPa·s/m²
        case GroundType::GRASS:        return   300.0; //    300 kPa·s/m²
    }
    return 300.0;
}

// ═════════════════════════════════════════════════════════════════════════════
// Sous-modèles de propagation
// ═════════════════════════════════════════════════════════════════════════════

// ─── Divergence géométrique ───────────────────────────────────────────────────

double AcousticModel::geometricalSpreading(double distance_m)
{
    // Plancher à 1 mm pour éviter log(0) et des valeurs infinies
    distance_m = std::max(distance_m, 1e-3);
    // Source ponctuelle : 20·log₁₀(r) + 10·log₁₀(4π)
    return 20.0 * std::log10(distance_m) + 10.0 * std::log10(4.0 * M_PI);
}

// ─── Absorption atmosphérique (ISO 9613-1) ────────────────────────────────────

double AcousticModel::absorptionCoefficient(double freq_Hz) const
{
    const double T     = params_.temperature_C + 273.15;   // [K]
    const double T_ref = 293.15;                            // 20 °C [K]
    const double T_rel = T / T_ref;
    const double p_rel = params_.pressure_kPa / 101.325;   // pression normalisée

    const double h = params_.humidity_pct;

    // Concentration molaire de vapeur d'eau
    double C     = -6.8346 * std::pow(273.16 / T, 1.261) + 4.6151;
    double h_mol = h * std::pow(10.0, C) * p_rel;

    // Fréquences de relaxation de l'oxygène et de l'azote
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

// ─── Angle d'élévation ────────────────────────────────────────────────────────

double AcousticModel::elevationAngle(double horiz_m, double dz_m)
{
    if (horiz_m < 1e-9 && std::abs(dz_m) < 1e-9) return 0.0;
    // θ vue depuis la source vers le récepteur, mesurée depuis l'horizontale
    // dz_m = h_source − h_receiver  → récepteur en-dessous = θ négatif
    double theta = std::atan2(-dz_m, horiz_m) * (180.0 / M_PI);
    return std::clamp(theta, -90.0, 90.0);
}

// ─── Correction de directivité (Eq. 1 + filtre high-shelving 500 Hz) ──────────

double AcousticModel::directivityCorrection(double theta_deg, double freq_Hz)
{
    // Polynôme de l'Eq. (1), normalisé à 0 dB pour |θ| = 30°
    double abs_theta = std::min(std::abs(theta_deg), 90.0);
    double G = -0.0011 * theta_deg * theta_deg
               + 0.194 * abs_theta
               - 4.9;

    // Filtre high-shelving : l'effet directif est réduit sous 500 Hz
    // (modélise le filtre décrit Section 2.2, fc=500 Hz, Q=0.5)
    if (freq_Hz < 500.0)
        G *= freq_Hz / 500.0;

    return G;  // [dB]
}

// ─── Effet de sol (Delany-Bazley + interférences) ────────────────────────────

double AcousticModel::groundEffect(double freq_Hz, double distance_m) const
{
    if (distance_m <= 0.0) return 0.0;

    const double hs = params_.source_height;
    const double hr = params_.receiver_height;
    const double dz = hs - hr;

    double d_horiz  = std::sqrt(std::max(0.0, distance_m * distance_m - dz * dz));
    double d_direct = distance_m;
    double d_reflect= std::sqrt(d_horiz * d_horiz + (hs + hr) * (hs + hr));
    double delta_d  = d_reflect - d_direct;

    // Impédance normalisée Delany-Bazley
    double f_over_sigma = freq_Hz / ground_resistivity_;
    f_over_sigma = std::max(f_over_sigma, 1e-10);

    double Z_real = 1.0 + 9.08  * std::pow(f_over_sigma, -0.75);
    double Z_imag = -11.9 * std::pow(f_over_sigma, -0.73);
    std::complex<double> Z_norm(Z_real, Z_imag);

    // Coefficient de réflexion onde plane
    double sin_psi = std::clamp((hs + hr) / d_reflect, 0.0, 1.0);
    std::complex<double> Zsin = Z_norm * sin_psi;
    std::complex<double> R_p  = (Zsin - 1.0) / (Zsin + 1.0);

    double Q_mag     = std::abs(R_p);
    double amp_ratio = Q_mag * (d_direct / d_reflect);

    // Interférences directe + réfléchie
    double k           = 2.0 * M_PI * freq_Hz / 343.0;
    double phase_refl  = std::arg(R_p);
    double total_phase = k * delta_d + phase_refl;

    double interference = 1.0
        + amp_ratio * amp_ratio
        + 2.0 * amp_ratio * std::cos(total_phase);

    interference = std::max(interference, 1e-10);
    return 10.0 * std::log10(interference);  // [dB]
}

// ═════════════════════════════════════════════════════════════════════════════
// Modèles spécifiques aux drones (Sections 2.3, 3.1, 3.2)
// ═════════════════════════════════════════════════════════════════════════════

// ─── Égalisation RPM (Eq. 2) ─────────────────────────────────────────────────

double AcousticModel::rpmEqualization(double freq_Hz, double rpm_actual,
                                      const DroneEmissionModel& model)
{
    // Recherche de la bande correspondante
    int band_idx = -1;
    double best  = std::numeric_limits<double>::max();
    for (int i = 0; i < NUM_BANDS; ++i) {
        double diff = std::abs(THIRD_OCTAVE_FREQS[i] - freq_Hz);
        if (diff < best) { best = diff; band_idx = i; }
    }

    // E(i) = S(i) * (R_actual − R_ref)   [Eq. 2]
    double S = model.equalizer_slope[band_idx];
    return S * (rpm_actual - model.rpm_ref);  // [dB]
}

// ─── Paramètres de manœuvre (Tableaux 4 & 5) ─────────────────────────────────

ManoeuvreParams AcousticModel::getManoeuvreParams(FlightManoeuvre manoeuvre,
                                                   const DroneEmissionModel& model)
{
    /**
     * RPM moyens tirés du Tableau 4 (valeurs pour M2).
     * Pour les autres drones, un facteur proportionnel au R_ref est appliqué.
     * Coefficients σn tirés du Tableau 5.
     */
    struct RawParams { double rpm_ratio; double a; double b; };

    // rpm_ratio = RPM_manoeuvre / RPM_ref (normalisé sur M2 R_ref=6540)
    static const std::unordered_map<int, RawParams> TABLE = {
        { (int)FlightManoeuvre::HOVER_10M,         { 6540.0/6540.0, 0.005, 0.0052 } },
        { (int)FlightManoeuvre::HOVER_20M,         { 6540.0/6540.0, 0.005, 0.0079 } },
        { (int)FlightManoeuvre::HOVER_50M,         { 6540.0/6540.0, 0.006, 0.0109 } },
        { (int)FlightManoeuvre::CLIMB,             { 6840.0/6540.0, 0.001, 0.0084 } },
        { (int)FlightManoeuvre::SINK,              { 6240.0/6540.0, 0.063, 0.0017 } },
        { (int)FlightManoeuvre::FORWARD_DOWNWIND,  { 6600.0/6540.0, 0.012, 0.0033 } },
        { (int)FlightManoeuvre::FORWARD_UPWIND,    { 6900.0/6540.0, 0.005, 0.0130 } },
    };

    auto it = TABLE.find((int)manoeuvre);
    if (it == TABLE.end())
        return { model.rpm_ref, 0.005, 0.0052 };

    const RawParams& rp = it->second;

    ManoeuvreParams mp;
    // RPM moyen = R_ref du drone × ratio de la manœuvre
    mp.rpm_average = model.rpm_ref * rp.rpm_ratio;
    mp.sigma_a     = rp.a;
    mp.sigma_b     = rp.b;
    return mp;
}

// ─── Déviation standard normalisée du RPM (Eq. 4) ────────────────────────────

double AcousticModel::rpmNormalisedStdDev(FlightManoeuvre manoeuvre,
                                           double wind_speed_ms,
                                           const DroneEmissionModel& model)
{
    ManoeuvreParams mp = getManoeuvreParams(manoeuvre, model);
    // σn = a + b * |v_wind|   [Eq. 4]
    return mp.sigma_a + mp.sigma_b * std::abs(wind_speed_ms);
}

// ─── Calcul du Lw effectif par bande ─────────────────────────────────────────

void AcousticModel::computeDroneLw(const DroneEmissionModel& model,
                                    double rpm_actual,
                                    double out_Lw[NUM_BANDS],
                                    double d_ref)
{
    // Lw(i) = Lp_ref(i) + E(i) + 20·log₁₀(d_ref) + 10·log₁₀(4π)
    //
    //  • Lp_ref(i) : niveau de référence en chambre anéchoïque @ d_ref, −30°
    //  • E(i)      : correction due au changement de RPM [Eq. 2]
    //  • Les deux derniers termes convertissent Lp @ d_ref → Lw (source ponctuelle)

    double geo_ref = 20.0 * std::log10(d_ref)
                   + 10.0 * std::log10(4.0 * M_PI);

    for (int i = 0; i < NUM_BANDS; ++i) {
        double E_i = model.equalizer_slope[i] * (rpm_actual - model.rpm_ref);
        out_Lw[i]  = model.Lp_ref[i] + E_i + geo_ref;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Réflexion au sol seule (source-image)
// ═════════════════════════════════════════════════════════════════════════════

double AcousticModel::computeReflectedSPL(double d_horiz, double hs, double hr) const
{
    // Distance du trajet réfléchi (méthode source-image)
    // Quand hs et hr sont nuls (source au sol), le trajet réfléchi dégénère
    // vers le trajet direct : d_reflected ≈ d_horiz
    double d_reflected = std::sqrt(d_horiz * d_horiz + (hs + hr) * (hs + hr));
    if (d_reflected < 1e-3)
        return -std::numeric_limits<double>::infinity();

    // Divergence géométrique sur le trajet réfléchi
    double A_div = geometricalSpreading(d_reflected);

    // Angle rasant au point de réflexion
    double sin_psi = std::clamp((hs + hr) / d_reflected, 0.0, 1.0);

    // ── Niveaux de puissance Lw ──────────────────────────────────────────────
    double Lw[NUM_BANDS];

    if (params_.drone_model != nullptr) {
        const DroneEmissionModel& model = *params_.drone_model;
        double rpm_actual = params_.rpm_actual;
        if (rpm_actual <= 0.0) {
            ManoeuvreParams mp = getManoeuvreParams(params_.manoeuvre, model);
            rpm_actual = mp.rpm_average;
        }
        computeDroneLw(model, rpm_actual, Lw, params_.d_ref);
    } else {
        for (int i = 0; i < NUM_BANDS; ++i)
            Lw[i] = params_.source_Lw[i];
    }

    // ── Calcul bande par bande ───────────────────────────────────────────────
    double sum = 0.0;
    for (int i = 0; i < NUM_BANDS; ++i) {
        double f = THIRD_OCTAVE_FREQS[i];

        // Absorption atmosphérique sur le trajet réfléchi
        double A_atm = atmosphericAbsorption(f, d_reflected);

        // Coefficient de réflexion Delany-Bazley
        double f_over_sigma = std::max(f / ground_resistivity_, 1e-10);
        double Z_real = 1.0 + 9.08  * std::pow(f_over_sigma, -0.75);
        double Z_imag = -11.9 * std::pow(f_over_sigma, -0.73);
        std::complex<double> Z_norm(Z_real, Z_imag);
        std::complex<double> Zsin  = Z_norm * sin_psi;
        std::complex<double> R_p   = (Zsin - 1.0) / (Zsin + 1.0);

        // Perte par réflexion [dB]
        double R_loss = 20.0 * std::log10(std::max(std::abs(R_p), 1e-10));

        // Lp = Lw − divergence − absorption_atm + réflexion
        double Lp = Lw[i] - A_div - A_atm + R_loss;

        // Sommation énergétique pondérée A
        double L_Aw = Lp + A_WEIGHTING[i];
        sum += std::pow(10.0, L_Aw / 10.0);
    }

    return 10.0 * std::log10(std::max(sum, 1e-30));
}

// ═════════════════════════════════════════════════════════════════════════════
// Calcul SPL principal
// ═════════════════════════════════════════════════════════════════════════════

void AcousticModel::computeSPLSpectrum(double distance_m, bool visible,
                                        double out_bands[NUM_BANDS]) const
{
    if (!visible || distance_m <= 0.0) {
        for (int i = 0; i < NUM_BANDS; ++i)
            out_bands[i] = -std::numeric_limits<double>::infinity();
        return;
    }

    // ── Géométrie ────────────────────────────────────────────────────────────
    const double dz      = params_.source_height - params_.receiver_height;
    const double d_horiz = std::sqrt(std::max(0.0,
                           distance_m * distance_m - dz * dz));
    const double theta   = elevationAngle(d_horiz, dz);
    const double A_div   = geometricalSpreading(distance_m);

    // ── Niveaux de puissance Lw ───────────────────────────────────────────────
    double Lw[NUM_BANDS];

    if (params_.drone_model != nullptr) {
        // Chemin principal : modèle drone complet
        const DroneEmissionModel& model = *params_.drone_model;

        // RPM effectif : valeur fournie ou RPM moyen de la manœuvre
        double rpm_actual = params_.rpm_actual;
        if (rpm_actual <= 0.0) {
            ManoeuvreParams mp = getManoeuvreParams(params_.manoeuvre, model);
            rpm_actual = mp.rpm_average;
        }

        computeDroneLw(model, rpm_actual, Lw, params_.d_ref);

    } else {
        // Chemin de repli : Lw fournis directement
        for (int i = 0; i < NUM_BANDS; ++i)
            Lw[i] = params_.source_Lw[i];
    }

    // ── Calcul bande par bande ────────────────────────────────────────────────
    for (int i = 0; i < NUM_BANDS; ++i) {
        double f     = THIRD_OCTAVE_FREQS[i];
        double A_atm = atmosphericAbsorption(f, distance_m);
        double D     = directivityCorrection(theta, f);
        // Effet de sol (interférences directe+réfléchie) seulement si réflexion activée
        double G     = (params_.reflection_order >= 1)
                       ? groundEffect(f, distance_m) : 0.0;

        // Lp(i) = Lw(i) − A_div − A_atm + D + G
        out_bands[i] = Lw[i] - A_div - A_atm + D + G;
    }
}

double AcousticModel::computeSPL(double distance_m, bool visible) const
{
    if (!visible || distance_m <= 0.0)
        return -std::numeric_limits<double>::infinity();

    double bands[NUM_BANDS];
    computeSPLSpectrum(distance_m, visible, bands);

    // Sommation énergétique des bandes pondérées A
    double sum = 0.0;
    for (int i = 0; i < NUM_BANDS; ++i) {
        double L_Aw = bands[i] + A_WEIGHTING[i];
        sum += std::pow(10.0, L_Aw / 10.0);
    }

    return 10.0 * std::log10(std::max(sum, 1e-30));
}
